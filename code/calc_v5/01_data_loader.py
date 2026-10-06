"""
Module 01 (v2): Data Loader
============================
Re-loads raw vessel static / voyage legs / port reference data (READ-ONLY),
adding the intermediate-call column `moorPortCode` (observed midway mooring
ports per leg) and the chokepoint-crossing list `legCrossNodeList`, both
required by the v2 sequence-based intermediate bunkering sets (Eq. 10).

Outputs (version-isolated): 02_数据_output/processed_v2/
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    get_data_paths, PORT_CSV, PROCESSED_DIR, V2_PROCESSED_DIR, PROCESSING_PARAMS,
    EMISSION_PARAMS, VESSEL_TYPE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 1. Vessel Static Data
# ============================================================
def load_vessel_static(vessel_csv: Path, vessel_type: str) -> pd.DataFrame:
    """Load and standardize vessel static data (same schema as v1)."""
    logger.info(f"Loading vessel static data from {vessel_csv}")
    df = pd.read_csv(vessel_csv, low_memory=False)
    logger.info(f"  Raw records: {len(df):,}")

    col_map = {
        "mmsi": "mmsi", "imo": "imo",
        "vesselNameEn": "vessel_name", "vesselNameCn": "vessel_name_cn",
        "flagCtryNameEn": "flag_country", "buildYear": "build_year",
        "dwt": "dwt", "grt": "grt", "teu": "teu",
        "length": "length_m", "width": "width_m", "draught": "draught_m",
        "speed": "design_speed_kn",
        "totalkilowattsofmainengines": "power_kw_total",
        "mainenginemodel": "engine_model", "mainenginedesigner": "engine_designer",
        "mainenginenumberofcylinders": "engine_cylinders",
        "vesselSub2TypeNameEn": "vessel_size_class",
        "operatorBodyCn": "operator", "ownerBodyCn": "owner",
        "ciiClass": "cii_class",
    }
    existing_cols = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=existing_cols)
    keep_cols = [v for v in existing_cols.values() if v in df.columns]
    df = df[keep_cols].copy()

    if "power_kw_total" in df.columns:
        df["p_me_75_kw"] = df["power_kw_total"] * EMISSION_PARAMS["mcr_fraction"]
    else:
        logger.warning("  No engine power column found! Will need imputation.")
        df["p_me_75_kw"] = np.nan

    df["vessel_type"] = vessel_type
    df["mmsi"] = pd.to_numeric(df["mmsi"], errors="coerce")
    df = df.dropna(subset=["mmsi"])
    df["mmsi"] = df["mmsi"].astype(int)
    df = df.drop_duplicates(subset=["mmsi"], keep="first")

    # ---- v2: class-median imputation for missing engine power / design speed
    # (manuscript: "Missing individual records are replaced with vessel-class
    # medians (grouped by TEU bracket)") — maximizes leg coverage with the
    # existing static table, no external data needed.
    if "teu" not in df.columns:
        df["teu"] = np.nan
    df["teu"] = pd.to_numeric(df["teu"], errors="coerce")
    teu_bins = [-np.inf, 500, 1000, 2000, 4000, 8000, 12000, np.inf]
    teu_labels = ["<500", "500-1k", "1k-2k", "2k-4k", "4k-8k", "8k-12k", ">12k"]
    df["teu_bracket"] = pd.cut(df["teu"], bins=teu_bins, labels=teu_labels)
    df["teu_bracket"] = df["teu_bracket"].cat.add_categories(["unknown"])
    df.loc[df["teu_bracket"].isna(), "teu_bracket"] = "unknown"

    for col in ["p_me_75_kw", "design_speed_kn"]:
        if col in df.columns:
            med = df.groupby("teu_bracket", observed=True)[col].transform("median")
            fallback = df[col].median()
            df[col] = df[col].fillna(med).fillna(fallback)
            logger.info(f"  Imputed {col}: missing filled with TEU-bracket medians")

    n_imp_p = int(df["p_me_75_kw"].isna().sum()) if "p_me_75_kw" in df.columns else 0
    logger.info(f"  Clean vessels: {len(df):,} (power still missing: {n_imp_p:,})")
    return df


# ============================================================
# 2. Voyage Leg Data (v2: + moorPortCode, legCrossNodeList)
# ============================================================
def load_voyage_legs(leg_dir: Path, sample_n: Optional[int] = None) -> pd.DataFrame:
    """Load all voyage leg CSV files into a single DataFrame (v2 schema)."""
    logger.info(f"Loading voyage legs from {leg_dir}")
    csv_files = sorted(leg_dir.glob("*.csv"))
    logger.info(f"  Found {len(csv_files):,} vessel leg files")

    if sample_n:
        csv_files = csv_files[:sample_n]
        logger.info(f"  Sampling first {sample_n} files")

    use_cols = [
        "mmsi", "legStartTime", "legEndTime",
        "legStartPortCode", "legStartPortNameEn", "legStartPortCtryNameEn",
        "legEndPortCode", "legEndPortNameEn", "legEndPortCtryNameEn",
        "arrivalTime", "departureTime",
        "sailDistance", "averageSpeed",
        "legDurationTotal", "sailDuration", "moorDuration",
        "isCanal", "isDirect", "isFullLoad", "loadType",
        "startLon", "startLat", "endLon", "endLat",
        "startDraught", "endDraught",
        "isAisAbnormal", "maxLostDuration",
        "startBerthTypeNameEn", "endBerthTypeNameEn",
        # v2 additions — observed midway mooring ports & chokepoint crossings
        "moorPortCode", "moorTimesPort", "moorTimesHalfway",
        "legCrossNodeList",
    ]

    chunks = []
    errors = 0
    for i, f in enumerate(csv_files):
        try:
            df_tmp = pd.read_csv(f, low_memory=False, nrows=0)
            available = [c for c in use_cols if c in df_tmp.columns]
            df_tmp = pd.read_csv(f, usecols=available, low_memory=False)
            if len(df_tmp) > 0:
                chunks.append(df_tmp)
        except Exception as e:
            errors += 1
            if errors <= 5:
                logger.warning(f"  Error reading {f.name}: {e}")
        if (i + 1) % 1000 == 0:
            logger.info(f"  Processed {i+1}/{len(csv_files)} files...")

    logger.info(f"  Read errors: {errors}")
    if not chunks:
        raise RuntimeError("No voyage leg data loaded!")

    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"  Total leg records: {len(df):,}")

    rename_map = {
        "legStartTime": "leg_start_time",
        "legEndTime": "leg_end_time",
        "legStartPortCode": "origin_port",
        "legStartPortNameEn": "origin_port_name",
        "legStartPortCtryNameEn": "origin_country",
        "legEndPortCode": "dest_port",
        "legEndPortNameEn": "dest_port_name",
        "legEndPortCtryNameEn": "dest_country",
        "arrivalTime": "arrival_time",
        "departureTime": "departure_time",
        "sailDistance": "sail_distance_nm",
        "averageSpeed": "avg_speed_kn",
        "legDurationTotal": "leg_duration_h",
        "sailDuration": "sail_duration_h",
        "moorDuration": "moor_duration_h",
        "isCanal": "is_canal",
        "isDirect": "is_direct",
        "isFullLoad": "is_full_load",
        "loadType": "load_type",
        "startLon": "origin_lon",
        "startLat": "origin_lat",
        "endLon": "dest_lon",
        "endLat": "dest_lat",
        "startDraught": "start_draught_m",
        "endDraught": "end_draught_m",
        "isAisAbnormal": "is_ais_abnormal",
        "maxLostDuration": "max_lost_duration_h",
        "startBerthTypeNameEn": "origin_berth_type",
        "endBerthTypeNameEn": "dest_berth_type",
        "moorPortCode": "moor_port_code",          # v2
        "moorTimesPort": "moor_times_port",        # v2
        "moorTimesHalfway": "moor_times_halfway",  # v2
        "legCrossNodeList": "leg_cross_nodes",     # v2
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    df["mmsi"] = pd.to_numeric(df["mmsi"], errors="coerce")
    for col in ["leg_start_time", "leg_end_time", "arrival_time", "departure_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["sail_distance_nm", "avg_speed_kn", "sail_duration_h",
                "start_draught_m", "end_draught_m"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter: keep only legs with actual sailing
    if "sail_distance_nm" in df.columns:
        n_before = len(df)
        df = df[df["sail_distance_nm"] > 0].copy()
        logger.info(f"  Filtered zero-distance legs: {n_before - len(df):,} removed")

    return df


# ============================================================
# 3. Port Reference Data
# ============================================================
def load_ports(port_csv: Path) -> pd.DataFrame:
    """Load global port reference data (same schema as v1)."""
    logger.info(f"Loading port data from {port_csv}")
    df = pd.read_csv(port_csv, encoding="utf-8")
    df = df.rename(columns={
        "portCode": "port_code",
        "nameCn": "port_name_cn",
        "nameEn": "port_name_en",
        "ctryCode": "country_code",
        "ctryNameCn": "country_cn",
        "ctryNameEn": "country_en",
        "lon": "port_lon",
        "lat": "port_lat",
        "continentNameEn": "continent",
    })
    keep = ["port_code", "port_name_cn", "port_name_en",
            "country_code", "country_cn", "country_en",
            "port_lon", "port_lat", "continent"]
    df = df[[c for c in keep if c in df.columns]].copy()
    logger.info(f"  Ports loaded: {len(df):,}")
    return df


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="GSC Data Loader v5")
    parser.add_argument("--vessel-type", default=VESSEL_TYPE,
                        choices=["container", "tanker"])
    parser.add_argument("--sample", type=int, default=None,
                        help="Only load first N vessel leg files (for testing)")
    parser.add_argument("--reuse-v2", action="store_true",
                        help="Copy the model-free v2 cleaned tables (vessels/legs/"
                             "ports) from processed_v2 into processed_v5 instead of "
                             "re-reading the provider CSVs (module 01 has no model "
                             "assumptions, so the tables are identical)")
    args = parser.parse_args()

    vessel_type = args.vessel_type

    if args.reuse_v2:
        names = [f"vessels_{vessel_type}.parquet", f"legs_{vessel_type}.parquet",
                 "ports_global.parquet"]
        for n in names:
            src = V2_PROCESSED_DIR / n
            dst = PROCESSED_DIR / n
            if not src.exists():
                raise FileNotFoundError(f"--reuse-v2: missing {src}")
            pd.read_parquet(src).to_parquet(dst, index=False)
            logger.info(f"Copied (reuse-v2): {src} -> {dst}")
        logger.info("Done (reuse-v2).")
        return

    paths = get_data_paths(vessel_type)
    logger.info(f"Pipeline v2 started for: {paths['label_cn']} ({paths['label']})")

    vessels = load_vessel_static(paths["vessel_csv"], vessel_type)
    legs = load_voyage_legs(paths["leg_dir"], sample_n=args.sample)
    ports = load_ports(PORT_CSV)

    # Match check
    leg_mmsis = set(legs["mmsi"].unique())
    vessel_mmsis = set(vessels["mmsi"].unique())
    logger.info(f"  Legs vessels ∩ Static vessels: {len(leg_mmsis & vessel_mmsis):,} "
                f"/ {len(leg_mmsis):,}")

    # v2: intermediate-call coverage report
    if "moor_port_code" in legs.columns:
        n_moor = legs["moor_port_code"].notna().sum()
        logger.info(f"  Legs with midway mooring port: {n_moor:,} "
                    f"({n_moor / len(legs) * 100:.1f}%)")
        logger.info(f"  Unique midway mooring ports: "
                    f"{legs['moor_port_code'].dropna().nunique():,}")

    # Save to versioned parquet
    out_vessels = PROCESSED_DIR / f"vessels_{vessel_type}.parquet"
    out_legs = PROCESSED_DIR / f"legs_{vessel_type}.parquet"
    out_ports = PROCESSED_DIR / "ports_global.parquet"

    vessels.to_parquet(out_vessels, index=False)
    legs.to_parquet(out_legs, index=False)
    ports.to_parquet(out_ports, index=False)

    logger.info(f"\nSaved (v2):")
    logger.info(f"  {out_vessels} ({out_vessels.stat().st_size / 1e6:.1f} MB)")
    logger.info(f"  {out_legs} ({out_legs.stat().st_size / 1e6:.1f} MB)")
    logger.info(f"  {out_ports} ({out_ports.stat().st_size / 1e6:.1f} MB)")
    logger.info("Done.")


if __name__ == "__main__":
    main()
