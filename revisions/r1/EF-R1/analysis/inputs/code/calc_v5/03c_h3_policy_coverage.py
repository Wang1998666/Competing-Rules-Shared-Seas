"""
Module 03c (v2): H3 Policy Coverage (EU ETS Maritime)
======================================================
Maps EU ETS Maritime jurisdiction onto H3 cells:
  * EEA port buffer zones (200 nm)
  * EEA-to-EEA voyage segments (full coverage, weight 1.0)
  * EEA-to-other voyage segments (50% coverage, weight 0.5)

v2: EU/EEA country set is unified with Module 12 (French outermost regions
excluded from the corridor-rule definition; see review finding on the
overlap between 03c and 12).

Also computes per-corridor PGI (Policy Gap Index) at H3 precision.

Outputs (02_数据_output/h3_grid_v2/):
  h3_eu_ets_coverage_{vt}.parquet, corridor_pgi_h3_{vt}.csv
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import h3

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PROCESSED_DIR, H3_DIR, H3_PARAMS, VESSEL_TYPE, get_data_paths,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# EU/EEA country codes — unified with Module 12 (no French overseas in the
# corridor-rule definition; they would misclassify open-ocean corridors).
EU_EEA_COUNTRIES = {
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
    "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
    "PL", "PT", "RO", "SK", "SI", "ES", "SE",
    "IS", "LI", "NO",  # EEA
}

PORT_BUFFER_NM = 200.0


def get_h3_ring_buffer(lat, lon, radius_nm, res=5):
    center = h3.latlng_to_cell(lat, lon, res)
    cell_edge_km = 8.5 if res == 5 else 1.3
    radius_km = radius_nm * 1.852
    n_rings = max(1, min(int(radius_km / cell_edge_km), 40))
    return list(h3.grid_disk(center, n_rings))


def build_eu_ets_h3_coverage(vessel_type: str):
    paths = get_data_paths(vessel_type)
    logger.info(f"Building EU ETS H3 coverage v2: {paths['label_cn']}")

    ports = pd.read_parquet(PROCESSED_DIR / "ports_global.parquet")
    ports_dedup = ports.drop_duplicates("port_code")

    country_col = None
    for col in ["country_code", "ctry_code", "ctryCode"]:
        if col in ports_dedup.columns:
            country_col = col
            break
    if country_col is None:
        logger.warning("  No country code column — coordinate-based EU detection")
        ports_dedup["is_eu"] = (
            (ports_dedup["port_lon"] >= -25) & (ports_dedup["port_lon"] <= 35) &
            (ports_dedup["port_lat"] >= 35) & (ports_dedup["port_lat"] <= 72))
    else:
        ports_dedup["is_eu"] = ports_dedup[country_col].isin(EU_EEA_COUNTRIES)

    eu_ports = ports_dedup[ports_dedup["is_eu"]].dropna(subset=["port_lon", "port_lat"])
    logger.info(f"  EU/EEA ports: {len(eu_ports)}")

    # ---- Step 1: port buffer zones ----
    res = H3_PARAMS["res_ocean"]
    buffer_cells = set()
    for _, row in eu_ports.iterrows():
        buffer_cells.update(get_h3_ring_buffer(row["port_lat"], row["port_lon"],
                                               PORT_BUFFER_NM, res))
    logger.info(f"  Port buffer H3 cells: {len(buffer_cells):,}")

    # ---- Step 2: voyage segments ----
    paths_file = H3_DIR / f"corridor_h3_paths_{vessel_type}.parquet"
    corridors_file = PROCESSED_DIR / f"corridors_{vessel_type}.parquet"
    eea_eea_cells, eea_other_cells = set(), set()
    if paths_file.exists():
        h3_paths = pd.read_parquet(paths_file)
        corridors = pd.read_parquet(corridors_file)
        port_eu = ports_dedup.set_index("port_code")["is_eu"].to_dict()
        corridors["origin_eu"] = corridors["origin_port"].map(port_eu).fillna(False)
        corridors["dest_eu"] = corridors["dest_port"].map(port_eu).fillna(False)
        eea_eea = corridors[corridors["origin_eu"] & corridors["dest_eu"]]["corridor_id"]
        eea_other = corridors[corridors["origin_eu"] ^ corridors["dest_eu"]]["corridor_id"]
        eea_eea_cells = set(h3_paths[h3_paths["corridor_id"].isin(eea_eea)]["h3_id"].unique())
        eea_other_cells = set(h3_paths[h3_paths["corridor_id"].isin(eea_other)]["h3_id"].unique())
        logger.info(f"  EEA-EEA cells: {len(eea_eea_cells):,}; "
                    f"EEA-other cells: {len(eea_other_cells):,}")

    # ---- Combine ----
    all_cells = buffer_cells | eea_eea_cells | eea_other_cells
    records = []
    for cell in all_cells:
        lat, lon = h3.cell_to_latlng(cell)
        if cell in buffer_cells or cell in eea_eea_cells:
            records.append({"h3_id": cell, "lat": lat, "lon": lon,
                            "coverage_type": "full", "weight": 1.0})
        else:
            records.append({"h3_id": cell, "lat": lat, "lon": lon,
                            "coverage_type": "partial", "weight": 0.5})
    coverage_df = pd.DataFrame(records)
    logger.info(f"  Total EU ETS H3 cells: {len(coverage_df):,} "
                f"(full {(coverage_df['weight']==1.0).sum():,}, "
                f"partial {(coverage_df['weight']==0.5).sum():,})")

    # ---- Per-corridor PGI ----
    pgi_df = pd.DataFrame()
    if paths_file.exists():
        h3_paths = pd.read_parquet(paths_file)
        eu_weight = coverage_df.set_index("h3_id")["weight"].to_dict()
        rows = []
        for cid, group in h3_paths.groupby("corridor_id"):
            cells = group["h3_id"].tolist()
            n_total = len(cells)
            if n_total == 0:
                continue
            eu_cov = sum(eu_weight.get(c, 0) for c in cells)
            rows.append({"corridor_id": cid, "PGI_h3": 1.0 - eu_cov / n_total,
                         "n_h3_cells": n_total, "eu_covered_cells": eu_cov})
        pgi_df = pd.DataFrame(rows)
        pgi_df.to_csv(H3_DIR / f"corridor_pgi_h3_{vessel_type}.csv", index=False)
        logger.info(f"  PGI: mean={pgi_df['PGI_h3'].mean():.3f}, "
                    f"fully covered: {(pgi_df['PGI_h3']==0).sum():,}, "
                    f"zero coverage: {(pgi_df['PGI_h3']>=0.99).sum():,}")

    coverage_df.to_parquet(H3_DIR / f"h3_eu_ets_coverage_{vessel_type}.parquet", index=False)
    logger.info("  Saved: h3_eu_ets_coverage (v2)")
    return coverage_df, pgi_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vessel-type", default=VESSEL_TYPE)
    args = parser.parse_args()
    build_eu_ets_h3_coverage(args.vessel_type)
