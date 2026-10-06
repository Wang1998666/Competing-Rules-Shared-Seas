"""
Module 03e (v4): Cell-Level Sea Filter for the Emission Grid
=============================================================
Root-fix post-processing (no 03 re-run needed): cells whose center falls on
land (0.25° mask) are NOT real visited cells — they come from AIS anomalies
(GPS drift / spoofed inland points), gap-interpolation chords across land,
or coarse coastal cells whose centroid lands on a land pixel. Following the
user directive ("cells not visited by AIS = land"), their emission is
RE-DISTRIBUTED to the nearest sea cell (grid-disk expansion), keeping total
emission exactly conserved while producing ZERO inland cells. The output
grid is therefore purely "AIS-visited sea cells".

Output: overwrites 02_数据_output/h3_grid_v4/h3_emission_grid_{vt}.parquet
"""

import sys
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import h3
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from config import H3_DIR, VESSEL_TYPE
import importlib
m03 = importlib.import_module("03_h3_attribution")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def clean_grid_cells(vessel_type: str):
    t0 = time.time()
    f = H3_DIR / f"h3_emission_grid_{vessel_type}.parquet"
    h3_df = pd.read_parquet(f)
    logger.info(f"  Loaded grid: {len(h3_df):,} cells, "
                f"{h3_df['emission_tco2e'].sum()/1e6:.2f} Mt")

    mask = m03._get_land_mask()
    latc = h3_df["h3_id"].apply(lambda h: h3.cell_to_latlng(h)[0]).values
    lonc = h3_df["h3_id"].apply(lambda h: h3.cell_to_latlng(h)[1]).values
    inland = m03._is_land_arr(lonc, latc, mask)

    n_inland = int(inland.sum())
    if n_inland == 0:
        logger.info("  No inland cells — grid already clean.")
        return h3_df

    inland_ids = h3_df.loc[inland, "h3_id"].tolist()
    inland_em = h3_df.loc[inland, "emission_tco2e"].values
    sea_ids = h3_df.loc[~inland, "h3_id"].tolist()
    logger.info(f"  Inland cells: {n_inland:,} ({inland_em.sum()/1e3:.1f} kt) — "
                f"re-distributing to nearest sea cells")

    # cKDTree on 3D unit-sphere coords of sea-cell centers (fast nearest sea)
    def to_xyz(lat, lon):
        lat_r, lon_r = np.radians(lat), np.radians(lon)
        return (np.cos(lat_r) * np.cos(lon_r),
                np.cos(lat_r) * np.sin(lon_r),
                np.sin(lat_r))

    sea_lat = h3_df.loc[~inland, "h3_id"].apply(
        lambda h: h3.cell_to_latlng(h)[0]).values
    sea_lon = h3_df.loc[~inland, "h3_id"].apply(
        lambda h: h3.cell_to_latlng(h)[1]).values
    sx, sy, sz = to_xyz(sea_lat, sea_lon)
    tree = cKDTree(np.column_stack([sx, sy, sz]))

    in_lat = h3_df.loc[inland, "h3_id"].apply(
        lambda h: h3.cell_to_latlng(h)[0]).values
    in_lon = h3_df.loc[inland, "h3_id"].apply(
        lambda h: h3.cell_to_latlng(h)[1]).values
    ix, iy, iz = to_xyz(in_lat, in_lon)
    dist, idx = tree.query(np.column_stack([ix, iy, iz]), k=1)

    # NOTE: nearest sea cell in 3D may be a far-away cell for deep-inland
    # anomalies; cap the distance to ~60 rings (~2300 km) — beyond that the
    # emission is dropped (statistical noise anyway) and reported.
    R = 6371.0
    d_km = 2 * R * np.arcsin(np.clip(dist / 2, 0, 1))
    ok = d_km <= 2300.0

    add = {}
    n_found = 0
    for i in range(len(inland_ids)):
        em = inland_em[i]
        if em <= 0 or not ok[i]:
            continue
        target = sea_ids[idx[i]]
        add[target] = add.get(target, 0.0) + em
        n_found += 1

    n_drop = int((~ok).sum())
    if n_drop:
        logger.warning(f"  {n_drop:,} inland cells beyond 2300 km — dropped "
                       f"({inland_em[~ok].sum()/1e3:.1f} kt)")
    logger.info(f"  Re-distributed {n_found:,}/{n_inland:,} inland cells "
                f"({time.time()-t0:.0f}s)")

    # build clean grid: sea cells + added emissions
    keep = ~inland
    out = h3_df.loc[keep].copy()
    if add:
        add_df = pd.DataFrame({"h3_id": list(add.keys()),
                               "emission_tco2e": list(add.values())})
        add_df = add_df.groupby("h3_id", as_index=False)["emission_tco2e"].sum()
        out = pd.concat([out, add_df], ignore_index=True)
        out = out.groupby("h3_id", as_index=False).agg(
            emission_tco2e=("emission_tco2e", "sum"),
            lat=("lat", "first"), lon=("lon", "first"),
            resolution=("resolution", "first"))
    out = out.sort_values("emission_tco2e", ascending=False).reset_index(drop=True)

    out.to_parquet(f, index=False)
    logger.info(f"  Clean grid saved: {len(out):,} cells "
                f"({out['emission_tco2e'].sum()/1e6:.2f} Mt, "
                f"conserved from {h3_df['emission_tco2e'].sum()/1e6:.2f} Mt)")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vessel-type", default=VESSEL_TYPE)
    args = parser.parse_args()
    clean_grid_cells(args.vessel_type)
