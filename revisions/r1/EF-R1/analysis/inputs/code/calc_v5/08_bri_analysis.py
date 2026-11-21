"""
Module 08 (v2): BRI Corridor Marking & Statistical Analysis
============================================================
Marks corridors by BRI maritime partner membership (endpoint country in the
59-state BRI set) and produces summary statistics.

Outputs (02_数据_output/policy_analysis_v2/):
  bri_corridors_{vt}.parquet, bri_port_summary_{vt}.csv, bri_summary_{vt}.json
"""

import sys
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PROCESSED_DIR, POLICY_DIR, VESSEL_TYPE, get_data_paths,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BRI_MARITIME_COUNTRIES = {
    # East Asia
    "CN", "HK", "MO", "TW",
    # Southeast Asia
    "VN", "TH", "MY", "SG", "ID", "PH", "MM", "KH", "BN", "TL",
    # South Asia
    "IN", "LK", "PK", "BD", "MV",
    # Middle East / West Asia
    "AE", "SA", "OM", "QA", "KW", "BH", "IR", "IQ", "YE", "JO", "IL", "TR",
    # East Africa
    "KE", "TZ", "DJ", "ET", "EG", "SD", "SO", "MZ", "MG",
    # Europe (Maritime BRI endpoints)
    "GR", "IT", "PT", "ES", "MT", "HR", "ME",
    # Oceania (Pacific BRI)
    "FJ", "PG", "WS", "TO", "VU", "SB",
    # Central Asia (land BRI but relevant ports via Caspian)
    "KZ", "UZ", "TM", "AZ", "GE",
    # Russia (Northern Sea Route BRI)
    "RU",
}


def bri_analysis(vessel_type: str):
    paths = get_data_paths(vessel_type)
    logger.info(f"BRI analysis v2: {paths['label_cn']}")

    corridors = pd.read_parquet(PROCESSED_DIR / f"corridors_{vessel_type}.parquet")
    ports = pd.read_parquet(PROCESSED_DIR / "ports_global.parquet")
    ports_dedup = ports.drop_duplicates("port_code", keep="first")
    country_of = ports_dedup.set_index("port_code")["country_code"].to_dict()

    def is_bri_port(p):
        return country_of.get(p, "") in BRI_MARITIME_COUNTRIES

    corridors["origin_bri"] = corridors["origin_port"].map(is_bri_port)
    corridors["dest_bri"] = corridors["dest_port"].map(is_bri_port)
    corridors["is_bri"] = corridors["origin_bri"] | corridors["dest_bri"]
    corridors["is_bri_full"] = corridors["origin_bri"] & corridors["dest_bri"]

    n_bri = corridors["is_bri"].sum()
    em_share = corridors.loc[corridors["is_bri"], "E_e"].sum() / corridors["E_e"].sum() * 100
    logger.info(f"  BRI corridors: {n_bri:,} ({n_bri/len(corridors)*100:.1f}%), "
                f"emission share: {em_share:.1f}%")

    out = corridors[["corridor_id", "origin_port", "dest_port", "E_e", "W_e",
                     "frequency", "is_bri", "is_bri_full"]].copy()
    out.to_parquet(POLICY_DIR / f"bri_corridors_{vessel_type}.parquet", index=False)

    # port-level summary
    port_codes = set(corridors["origin_port"]) | set(corridors["dest_port"])
    port_rows = []
    for p in port_codes:
        n_ep = int(((corridors["origin_port"] == p) | (corridors["dest_port"] == p)).sum())
        port_rows.append({"port_code": p, "country_code": country_of.get(p, ""),
                          "is_bri": is_bri_port(p),
                          "n_corridors": n_ep})
    port_df = pd.DataFrame(port_rows)
    port_df.to_csv(POLICY_DIR / f"bri_port_summary_{vessel_type}.csv", index=False)

    summary = {"version": "v2",
               "n_corridors": int(len(corridors)),
               "n_bri": int(n_bri),
               "bri_share_pct": round(n_bri / len(corridors) * 100, 1),
               "bri_emission_share_pct": round(em_share, 1),
               "n_bri_ports": int(port_df["is_bri"].sum())}
    with open(POLICY_DIR / f"bri_summary_{vessel_type}.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("  Saved: bri_corridors / bri_port_summary / bri_summary (v2)")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vessel-type", default=VESSEL_TYPE)
    args = parser.parse_args()
    bri_analysis(args.vessel_type)
