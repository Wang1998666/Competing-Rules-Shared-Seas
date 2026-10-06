"""
Module 03 (v4): H3 Grid Emission Attribution — Root-Fix Version
================================================================
v4 root fix: the great-circle fallback for legs without a matched AIS
segment used to draw straight lines ACROSS continents (e.g. Australia,
Central Asia), creating pseudo-inland cells that cover whole landmasses
in heatmaps. Two changes fix this at the source:
  1. Stronger trajectory matching: exact (origin,dest) marker groups are
     tried first; if absent, a TIME-WINDOW match on [leg_start, leg_end]
     (±2 h buffer) recovers the real path from the vessel's trajectory.
  2. Sea-constrained GC fallback: remaining fallback legs keep only their
     sea-side segments (emission re-scaled by kept length, still exactly
     conserved); legs entirely on land are attributed to the nearest
     sea cells of their endpoints (no inland cells are ever produced).

Spatial attribution still uses REAL AIS trajectories; emission NUMBERS
come from the voyage-leg table (Module 02) so conservation stays exact:
    E_h = Σ_leg E_odv · (path-length of leg inside h) / (leg path length)

Resolution: Res 5 open ocean, Res 7 chokepoints.
Parallelized over vessels; outputs h3_emission_grid_{vt}.parquet and
h3_stats_{vt}.json into 02_数据_output/h3_grid_v4/.
"""

import sys
import argparse
import json
import logging
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import h3
import h3.api.basic_int as h3i

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PROCESSED_DIR, H3_DIR, H3_PARAMS, PROCESSING_PARAMS,
    VESSEL_TYPE, get_data_paths,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

R_EARTH_KM = 6371.0088
KM_PER_NM = 1.852

_V_LATLNG_INT = np.vectorize(h3i.latlng_to_cell, otypes=[np.int64])

# Lazy per-process land mask (0.25°) for the sea-constrained fallback.
# Built once per worker process (Windows spawn re-imports the module).
_LAND_MASK = None
_LAND_STEP = 0.25


def _get_land_mask():
    global _LAND_MASK
    if _LAND_MASK is None:
        _LAND_MASK = _build_land_mask(_LAND_STEP)
    return _LAND_MASK


def _build_land_mask(step):
    import cartopy.io.shapereader as shpreader
    from shapely.geometry import Point
    n_lon = int(round(360 / step))
    n_lat = int(round(180 / step))
    mask = np.zeros((n_lon, n_lat), dtype=bool)
    shp = shpreader.natural_earth(resolution="110m", category="physical", name="land")
    for g in shpreader.Reader(shp).geometries():
        if g.is_empty:
            continue
        x0, y0, x1, y1 = g.bounds
        i0 = max(int((x0 + 180) / step), 0)
        i1 = min(int((x1 + 180) / step) + 1, n_lon)
        j0 = max(int((y0 + 90) / step), 0)
        j1 = min(int((y1 + 90) / step) + 1, n_lat)
        for i in range(i0, i1):
            lon = -180 + (i + 0.5) * step
            for j in range(j0, j1):
                lat = -90 + (j + 0.5) * step
                if g.contains(Point(lon, lat)):
                    mask[i, j] = True
    return mask


def _is_land_arr(lon, lat, mask=None, step=0.25):
    """Vectorized land test for lon/lat arrays."""
    if mask is None:
        mask = _get_land_mask()
    i = np.clip(((lon + 180) / step).astype(int), 0, mask.shape[0] - 1)
    j = np.clip(((lat + 90) / step).astype(int), 0, mask.shape[1] - 1)
    return mask[i, j]


def nearest_sea_cell(lat, lon, res=5):
    """Nearest cell (at given res) whose center is on sea; None if not found.
    Up to 60 rings (~2300 km) covers inland river ports."""
    mask = _get_land_mask()
    c = h3i.latlng_to_cell(lat, lon, res)
    if not _is_land_arr(np.array([h3i.cell_to_latlng(c)[1]]),
                        np.array([h3i.cell_to_latlng(c)[0]]), mask)[0]:
        return c
    for k in range(1, 60):
        for cc in h3i.grid_disk(c, k):
            lat_c, lon_c = h3i.cell_to_latlng(cc)
            if not _is_land_arr(np.array([lon_c]), np.array([lat_c]), mask)[0]:
                return cc
    return None


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def is_chokepoint_arr(lat, lon):
    mask = np.zeros(len(lat), dtype=bool)
    for region in H3_PARAMS["chokepoint_regions"].values():
        mask |= ((lat >= region["lat_range"][0]) & (lat <= region["lat_range"][1]) &
                 (lon >= region["lon_range"][0]) & (lon <= region["lon_range"][1]))
    return mask


def clean_interpolate(lat, lon, t):
    """Jump removal, stationary removal, gap interpolation (Eq. 1)."""
    idx = np.argsort(t)
    lat, lon, t = lat[idx], lon[idx], t[idx]
    # jump removal (>40 kn)
    dt_h = np.diff(t) / 3600.0
    with np.errstate(divide="ignore", invalid="ignore"):
        spd = haversine_km(lon[:-1], lat[:-1], lon[1:], lat[1:]) / np.maximum(dt_h, 1e-9)
    keep = np.ones(len(lat), dtype=bool)
    keep[1:] = spd <= PROCESSING_PARAMS["jump_speed_threshold_kn"]
    keep[:-1] &= spd <= PROCESSING_PARAMS["jump_speed_threshold_kn"]
    lat, lon, t = lat[keep], lon[keep], t[keep]
    if len(lat) < 2:
        return lat, lon
    # gap interpolation (>30 min @ 5 min)
    dt_min = np.diff(t) / 60.0
    gap = dt_min > PROCESSING_PARAMS["gap_interpolation_minutes"]
    out_lat, out_lon = [lat[:1]], [lon[:1]]
    for i in range(len(lat) - 1):
        out_lat.append(lat[i + 1:i + 2])
        out_lon.append(lon[i + 1:i + 2])
        if gap[i]:
            n = int(np.floor(dt_min[i] / PROCESSING_PARAMS["interpolation_step_minutes"]))
            if n > 0:
                frac = np.linspace(0, 1, n + 2)[1:-1]
                dlon = lon[i + 1] - lon[i]
                if dlon > 180:
                    dlon -= 360
                elif dlon < -180:
                    dlon += 360
                out_lat.append(lat[i] + frac * (lat[i + 1] - lat[i]))
                out_lon.append((lon[i] + frac * dlon + 180) % 360 - 180)
    return np.concatenate(out_lat), np.concatenate(out_lon)


def attribute_path(lat, lon, e_leg):
    """Distribute e_leg over H3 cells proportional to segment path length.
    Vectorized: cleaned points are already ~1-2 km apart, so each consecutive
    pair is a segment; its emission goes to the segment-midpoint cell. No
    resampling loop (that double-sampling was the v3 speed bottleneck)."""
    seg_len = haversine_km(lon[:-1], lat[:-1], lon[1:], lat[1:])
    total = seg_len.sum()
    if total <= 0 or e_leg <= 0:
        return np.array([], dtype=np.int64), np.array([])
    seg_em = e_leg * seg_len / total
    mid_lat = (lat[:-1] + lat[1:]) / 2
    mid_lon = (lon[:-1] + lon[1:]) / 2
    res = np.where(is_chokepoint_arr(mid_lat, mid_lon),
                   H3_PARAMS["res_chokepoint"], H3_PARAMS["res_ocean"])
    cells = _V_LATLNG_INT(mid_lat, mid_lon, res)
    uniq, inv = np.unique(cells, return_inverse=True)
    out = np.zeros(len(uniq))
    np.add.at(out, inv, seg_em)
    return uniq, out


def _leg_endpoints_plausible(row):
    """脏航段端点校验：坐标缺失、超出真实集装箱港口纬度范围（|lat| > 56，
    最南正常港口乌斯怀亚 -54.8°）或大圆距离超过 20,000 km 的航段视为损坏数据，
    禁止进入大圆回退（防穿陆铁律补充，2026-08-24：曾发现 dest_lat=-59.6（南极）
    的脏航段经 GC 回退在南大洋画出直线伪影）。"""
    if pd.isna(row.get("origin_lat")) or pd.isna(row.get("dest_lat")) or \
       pd.isna(row.get("origin_lon")) or pd.isna(row.get("dest_lon")):
        return False
    if abs(row["origin_lat"]) > 56 or abs(row["dest_lat"]) > 56:
        return False
    d = haversine_km(row["origin_lon"], row["origin_lat"],
                     row["dest_lon"], row["dest_lat"])
    return d <= 20000.0


def gc_fallback(o_lat, o_lon, d_lat, d_lon, e_leg):
    """Sea-constrained great-circle fallback (root fix): interpolate the GC
    path, keep ONLY sea-side segments and re-scale emission by kept length
    (exact conservation, no inland cells). If every segment is on land the
    leg is attributed half/half to the nearest sea cells of its endpoints."""
    d = haversine_km(o_lon, o_lat, d_lon, d_lat)
    n = max(2, int(np.ceil(d / H3_PARAMS["gc_interp_km"])))
    frac = np.linspace(0, 1, n)
    dlon = d_lon - o_lon
    if dlon > 180:
        dlon -= 360
    elif dlon < -180:
        dlon += 360
    lat = o_lat + frac * (d_lat - o_lat)
    lon = (o_lon + frac * dlon + 180) % 360 - 180
    mask = _get_land_mask()
    seg_len = haversine_km(lon[:-1], lat[:-1], lon[1:], lat[1:])
    mid_lat = (lat[:-1] + lat[1:]) / 2
    mid_lon = (lon[:-1] + lon[1:]) / 2
    sea = ~_is_land_arr(mid_lon, mid_lat, mask)
    keep_len = seg_len[sea].sum()
    if keep_len <= 0:
        # entirely over land (river legs): attribute to nearest sea cells of
        # the endpoints — half/half, still conserved, no inland cells.
        cells, emis = [], {}
        for (la, lo) in ((o_lat, o_lon), (d_lat, d_lon)):
            cc = nearest_sea_cell(la, lo)
            if cc is not None:
                emis[cc] = emis.get(cc, 0.0) + e_leg / 2.0
        return (np.array(list(emis.keys()), dtype=np.int64),
                np.array(list(emis.values()), dtype=float))
    seg_em = e_leg * seg_len / keep_len
    seg_em[~sea] = 0.0
    res = np.where(is_chokepoint_arr(mid_lat, mid_lon),
                   H3_PARAMS["res_chokepoint"], H3_PARAMS["res_ocean"])
    cells = _V_LATLNG_INT(mid_lat, mid_lon, res)
    uniq, inv = np.unique(cells, return_inverse=True)
    out = np.zeros(len(uniq))
    np.add.at(out, inv, seg_em)
    return uniq, out


def process_vessel(mmsi, traj_file, legs_sub):
    """Attribute all legs of one vessel using its real trajectory."""
    try:
        df = pd.read_csv(traj_file, usecols=["postime", "lon", "lat", "sog",
                                             "legStartPortCode", "legEndPortCode"],
                         dtype={"postime": str, "legStartPortCode": str,
                                "legEndPortCode": str})
    except Exception:
        df = pd.DataFrame()

    have_traj = len(df) > 1
    if have_traj:
        df = df.dropna(subset=["lon", "lat", "sog"])
        df = df[(df["lon"] >= -180) & (df["lon"] <= 180) &
                (df["lat"] >= -90) & (df["lat"] <= 90) & (df["sog"] >= 0)]
        df["postime"] = pd.to_datetime(df["postime"], errors="coerce")
        df = df.dropna(subset=["postime"])
        have_traj = len(df) > 1
    if have_traj:
        # group trajectory by leg markers
        traj_groups = {k: g for k, g in
                       df.groupby(["legStartPortCode", "legEndPortCode"], dropna=False)
                       if len(g) >= 2}

    recs = []
    n_gc = 0
    for _, row in legs_sub.iterrows():
        e_leg = row["emission_tco2e"]
        if not (e_leg > 0):
            continue
        o, d = str(row["origin_port"]), str(row["dest_port"])
        lat = lon = None
        if have_traj:
            g = traj_groups.get((o, d))
            if g is not None and len(g) >= 2:
                # pick the time-window matching this leg
                t0 = pd.to_datetime(row["leg_start_time"])
                if pd.notna(t0):
                    dt = (g["postime"] - t0).abs()
                    g = g.iloc[dt.argmin():] if False else g  # keep full group
                lat, lon = clean_interpolate(g["lat"].values, g["lon"].values,
                                             g["postime"].values.astype(
                                                 "datetime64[s]").astype(np.int64))
            elif True:  # exact (o,d) group missing — try time-window match
                # Root fix: time-window match on the whole trajectory when the
                # (o,d) marker group is missing (markers absent/inconsistent).
                t0 = pd.to_datetime(row["leg_start_time"])
                t1 = pd.to_datetime(row["leg_end_time"])
                if pd.notna(t0) and pd.notna(t1):
                    w = (df["postime"] >= t0 - pd.Timedelta(hours=2)) & \
                        (df["postime"] <= t1 + pd.Timedelta(hours=2))
                    gw = df.loc[w]
                    if len(gw) >= 2:
                        lat, lon = clean_interpolate(
                            gw["lat"].values, gw["lon"].values,
                            gw["postime"].values.astype(
                                "datetime64[s]").astype(np.int64))
        if lat is None or len(lat) < 2:
            if _leg_endpoints_plausible(row):
                cells, emis = gc_fallback(row["origin_lat"], row["origin_lon"],
                                          row["dest_lat"], row["dest_lon"], e_leg)
                n_gc += 1
                if len(cells):
                    recs.append(pd.DataFrame({"h3_int": cells, "emission_tco2e": emis}))
            else:
                # 脏航段（坐标缺失/极地虚假坐标/超长距离）：禁止大圆回退，
                # 直接丢弃该航段，避免在南大洋等无航线区域画出直线伪影。
                logger.warning(f"  MMSI {mmsi}: implausible leg endpoints "
                               f"({row.get('origin_port')}->{row.get('dest_port')}), "
                               f"GC fallback skipped ({e_leg/1e3:.2f} kt dropped)")
            continue
        cells, emis = attribute_path(lat, lon, e_leg)
        if len(cells):
            recs.append(pd.DataFrame({"h3_int": cells, "emission_tco2e": emis}))

    if not recs:
        return pd.DataFrame(columns=["h3_int", "emission_tco2e"])
    return pd.concat(recs, ignore_index=True)


def _wrapper(args):
    mmsi, traj_file, legs_sub = args
    try:
        r = process_vessel(mmsi, traj_file, legs_sub)
        if len(r):
            r = r.copy()
            r["mmsi"] = mmsi
        return r
    except Exception as e:
        logger.debug(f"  Error MMSI {mmsi}: {e}")
        return pd.DataFrame(columns=["h3_int", "emission_tco2e", "mmsi"])


def run_h3_attribution(vessel_type):
    paths = get_data_paths(vessel_type)
    logger.info(f"H3 Attribution v2 (trajectory-based, conserved): {paths['label_cn']}")

    emissions = pd.read_parquet(PROCESSED_DIR / f"emissions_{vessel_type}.parquet")
    emissions = emissions[emissions["emission_tco2e"] > 0].copy()
    total_leg = emissions["emission_tco2e"].sum()
    logger.info(f"  Legs: {len(emissions):,}, total {total_leg/1e6:.2f} Mt")

    traj_map = {}
    for traj_dir in paths["traj_dirs"]:
        if traj_dir.exists():
            for f in traj_dir.glob("*.csv"):
                traj_map[int(f.stem)] = str(f)
    logger.info(f"  Trajectory files: {len(traj_map):,}")

    # ---- Incremental save + resume: per-chunk part parquets ----
    PART_DIR = H3_DIR / f"parts_{vessel_type}"
    PART_DIR.mkdir(parents=True, exist_ok=True)
    done_mmsi = set()
    for pf in PART_DIR.glob("part_*.parquet"):
        try:
            done_mmsi |= set(pd.read_parquet(pf, columns=["mmsi"])["mmsi"].unique())
        except Exception:
            pass
    if done_mmsi:
        logger.info(f"  Resume: {len(done_mmsi):,} vessels already done")

    tasks = [(int(m), traj_map.get(int(m)), sub)
             for m, sub in emissions.groupby("mmsi") if int(m) not in done_mmsi]
    logger.info(f"  Vessels to process this run: {len(tasks):,}")

    n_workers = PROCESSING_PARAMS["n_workers"]
    SAVE_EVERY = 500
    t0 = time.time()
    buffer = []
    part_idx = len(list(PART_DIR.glob("part_*.parquet")))

    def flush():
        nonlocal buffer, part_idx
        if buffer:
            pd.concat(buffer, ignore_index=True).to_parquet(
                PART_DIR / f"part_{part_idx:04d}.parquet", index=False)
            part_idx += 1
            buffer = []

    if tasks:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(_wrapper, t) for t in tasks]
            for i, fut in enumerate(as_completed(futs)):
                r = fut.result()
                if len(r):
                    buffer.append(r)
                if (i + 1) % SAVE_EVERY == 0:
                    flush()
                    el = time.time() - t0
                    logger.info(f"  Progress {i+1}/{len(tasks)} ({el/60:.1f} min) [saved]")
        flush()
    logger.info(f"  Vessel pass done in {(time.time()-t0)/60:.1f} min")

    # ---- Aggregate all parts ----
    part_files = sorted(PART_DIR.glob("part_*.parquet"))
    if not part_files:
        raise RuntimeError("No H3 data produced")
    allr = pd.concat([pd.read_parquet(p) for p in part_files], ignore_index=True)
    agg = allr.groupby("h3_int")["emission_tco2e"].sum().reset_index()
    # Save raw int grid first (insurance) before string/lat-lon conversion.
    agg.to_parquet(H3_DIR / f"h3_emission_grid_int_{vessel_type}.parquet", index=False)
    # Use basic_int API (h3i) for int->str / lat-lng / resolution conversions.
    agg["h3_id"] = agg["h3_int"].apply(lambda c: h3i.int_to_str(int(c)))
    agg["lat"] = agg["h3_int"].apply(lambda c: h3i.cell_to_latlng(int(c))[0])
    agg["lon"] = agg["h3_int"].apply(lambda c: h3i.cell_to_latlng(int(c))[1])
    agg["resolution"] = agg["h3_int"].apply(lambda c: h3i.get_resolution(int(c)))
    agg = agg.sort_values("emission_tco2e", ascending=False).reset_index(drop=True)

    total_h3 = agg["emission_tco2e"].sum()
    logger.info(f"  H3 cells: {len(agg):,} (Res5 {(agg['resolution']==5).sum():,}, "
                f"Res7 {(agg['resolution']==7).sum():,})")
    logger.info(f"  Total H3: {total_h3/1e6:.2f} Mt | conservation {total_h3/total_leg:.4%}")

    agg[["h3_id", "lat", "lon", "resolution", "emission_tco2e"]].to_parquet(
        H3_DIR / f"h3_emission_grid_{vessel_type}.parquet", index=False)
    p75 = float(agg["emission_tco2e"].quantile(0.75))
    stats = {"version": "v2-trajectory", "n_cells": int(len(agg)),
             "total_Mt": round(total_h3 / 1e6, 2), "p75_t": p75,
             "conservation_ratio": float(total_h3 / total_leg),
             "n_res5": int((agg["resolution"] == 5).sum()),
             "n_res7": int((agg["resolution"] == 7).sum())}
    with open(H3_DIR / f"h3_stats_{vessel_type}.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"  Saved. P75={p75:.1f} t")
    return agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vessel-type", default=VESSEL_TYPE)
    args = parser.parse_args()
    run_h3_attribution(args.vessel_type)
