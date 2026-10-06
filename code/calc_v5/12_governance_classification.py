"""
Module 12 (v2): Governance Classification (Tri-Polar Rule Attribution)
======================================================================
Assigns every corridor to one of four mutually exclusive governance types
(Eq. 20–22) using v2 data:

  * EU ETS: endpoint rule (EEA-to-EEA w=1.0, EEA-to-other w=0.5), country set
    unified with Module 03c.
  * BRI: endpoint-country membership (Module 08 flags).
  * FuelEU: max pressure along the corridor's H3 path ≥ τ=0.30 (τ-sensitivity
    statistics reported).
  * Vacuum: RE = 0; RVI (Eq. 21) strict (high-emission) and any.

Port-level camp attribution + hub control shares (HCS/CCS) from the
Module 05 20%-budget methanol solution (symmetric endpoint attribution).

Outputs (02_数据_output/policy_analysis_v2/): governance_*_{vt}.*,
vacuum_corridors_{vt}.csv, hub_control_{vt}.csv, camp_control_{vt}.csv
"""

import sys
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PROCESSED_DIR, RESULTS_DIR, H3_DIR, POLICY_DIR, VESSEL_TYPE,
)
from importlib import import_module
_08 = import_module("08_bri_analysis")
_03c = import_module("03c_h3_policy_coverage")
BRI_MARITIME_COUNTRIES = _08.BRI_MARITIME_COUNTRIES
EU_EEA_COUNTRIES = _03c.EU_EEA_COUNTRIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FUELEU_PRESSURE_TAU = 0.30
FUELEU_TAU_SENSITIVITY = [0.10, 0.30, 0.50]


def classify_corridors(vessel_type: str):
    corridors = pd.read_parquet(PROCESSED_DIR / f"corridors_{vessel_type}.parquet")
    bri_df = pd.read_parquet(POLICY_DIR / f"bri_corridors_{vessel_type}.parquet")
    ports = pd.read_parquet(PROCESSED_DIR / "ports_global.parquet")
    pgi_file = H3_DIR / f"corridor_pgi_h3_{vessel_type}.csv"
    paths_file = H3_DIR / f"corridor_h3_paths_{vessel_type}.parquet"
    pressure_file = POLICY_DIR / f"fueleu_pressure_{vessel_type}.parquet"

    gov = corridors[["corridor_id", "E_e", "W_e", "origin_port", "dest_port"]].copy()

    # ---- BRI sphere (Module 08) ----
    bri_cols = bri_df[["corridor_id", "is_bri", "is_bri_full"]].drop_duplicates("corridor_id")
    gov = gov.merge(bri_cols, on="corridor_id", how="left")
    gov["is_bri"] = gov["is_bri"].fillna(False)
    gov["is_bri_full"] = gov["is_bri_full"].fillna(False)

    # ---- EU ETS sphere (endpoint rule, Eq. 20) ----
    port_eu = (ports.drop_duplicates("port_code").set_index("port_code")["country_code"]
               .isin(EU_EEA_COUNTRIES))
    gov["origin_eu"] = gov["origin_port"].map(port_eu).fillna(False)
    gov["dest_eu"] = gov["dest_port"].map(port_eu).fillna(False)
    gov["eu_ets_endpoint"] = gov["origin_eu"] | gov["dest_eu"]
    gov["eu_ets_full"] = gov["origin_eu"] & gov["dest_eu"]
    gov["eu_ets_share"] = 0.0
    if pgi_file.exists():
        pgi = pd.read_csv(pgi_file)
        pgi["eu_share"] = pgi["eu_covered_cells"] / pgi["n_h3_cells"].clip(lower=1)
        gov = gov.merge(pgi[["corridor_id", "eu_share"]], on="corridor_id", how="left")
        gov["eu_share"] = gov["eu_share"].fillna(0.0)

    # ---- FuelEU pressure sphere (Eq. 19 + threshold) ----
    gov["fueleu_max"] = 0.0
    gov["fueleu_exposed"] = False
    if paths_file.exists() and pressure_file.exists():
        paths_df = pd.read_parquet(paths_file)
        pressure_df = pd.read_parquet(pressure_file)
        pres = pressure_df[["h3_id", "fueleu_pressure"]]
        merged = paths_df[["corridor_id", "h3_id"]].merge(pres, on="h3_id", how="left")
        p_max = merged.groupby("corridor_id")["fueleu_pressure"].max()
        gov["fueleu_max"] = gov["corridor_id"].map(p_max).fillna(0.0)
        for tau in FUELEU_TAU_SENSITIVITY:
            logger.info(f"    FuelEU-exposed (P>={tau}): "
                        f"{(gov['fueleu_max'] >= tau).sum():,}")
        gov["fueleu_exposed"] = gov["fueleu_max"] >= FUELEU_PRESSURE_TAU

    # ---- Rule exposure & mutually exclusive type (Eq. 20) ----
    gov["RE"] = (gov["eu_ets_endpoint"].astype(int) + gov["is_bri"].astype(int) +
                 gov["fueleu_exposed"].astype(int))
    conditions = [
        gov["eu_ets_endpoint"],
        ~gov["eu_ets_endpoint"] & gov["is_bri"],
        ~gov["eu_ets_endpoint"] & ~gov["is_bri"] & gov["fueleu_exposed"],
        gov["RE"] == 0,
    ]
    gov["gov_type"] = np.select(conditions, ["EU_ETS", "BRI_only", "FuelEU_exposed", "Vacuum"],
                                default="Vacuum")

    # ---- Vacuum list & RVI (Eq. 21) ----
    p75 = gov.loc[gov["E_e"] > 0, "E_e"].quantile(0.75)
    gov["is_vacuum"] = (gov["gov_type"] == "Vacuum") & (gov["E_e"] >= p75)
    total_em = gov["E_e"].sum()
    rvi_strict = gov.loc[gov["is_vacuum"], "E_e"].sum() / total_em * 100.0
    rvi_any = gov.loc[gov["gov_type"] == "Vacuum", "E_e"].sum() / total_em * 100.0
    logger.info(f"  P75(E_e) = {p75:,.0f} t; high-emission vacuum: "
                f"{gov['is_vacuum'].sum():,} ({rvi_strict:.1f}% of emissions); "
                f"RVI(any) = {rvi_any:.1f}%")

    gov.to_parquet(POLICY_DIR / f"governance_corridors_{vessel_type}.parquet", index=False)
    vac_cols = ["corridor_id", "origin_port", "dest_port", "E_e", "W_e",
                "RE", "eu_ets_endpoint", "is_bri", "fueleu_exposed"]
    gov.loc[gov["is_vacuum"], vac_cols].sort_values("E_e", ascending=False).to_csv(
        POLICY_DIR / f"vacuum_corridors_{vessel_type}.csv", index=False)
    return gov, p75, rvi_strict, rvi_any


def classify_ports(vessel_type: str, gov: pd.DataFrame):
    ports = pd.read_parquet(PROCESSED_DIR / "ports_global.parquet")
    ports_dedup = ports.drop_duplicates("port_code").copy()
    ports_dedup["is_bri"] = ports_dedup["country_code"].isin(BRI_MARITIME_COUNTRIES)
    ports_dedup["is_eu"] = ports_dedup["country_code"].isin(EU_EEA_COUNTRIES)
    ports_dedup["camp"] = np.select(
        [ports_dedup["is_eu"], ports_dedup["is_bri"]], ["EU", "BRI"], default="Other")

    sel_file = RESULTS_DIR / f"selected_ports_{vessel_type}.csv"
    selected = set(pd.read_csv(sel_file)["port_code"]) if sel_file.exists() else set()
    ports_dedup["is_selected"] = ports_dedup["port_code"].isin(selected)

    act_file = RESULTS_DIR / f"activated_corridors_{vessel_type}.csv"
    activated = pd.read_csv(act_file)
    abate_col = "W_e" if "W_e" in activated.columns else "E_e"
    activated = activated[["corridor_id", "origin_port", "dest_port", abate_col]] \
        .rename(columns={abate_col: "abate"})

    # Hub control share — symmetric ENDPOINT attribution (Eq. 27 convention)
    port_corr_links = pd.concat([
        activated[["corridor_id", "origin_port", "abate"]].rename(
            columns={"origin_port": "port_code"}),
        activated[["corridor_id", "dest_port", "abate"]].rename(
            columns={"dest_port": "port_code"}),
    ])
    hub = port_corr_links.groupby("port_code")["abate"].sum().reset_index() \
        .rename(columns={"abate": "unlocked_abate"})
    n_corr = port_corr_links.groupby("port_code")["corridor_id"].nunique().reset_index() \
        .rename(columns={"corridor_id": "n_corridors_served"})
    hub = hub.merge(n_corr, on="port_code")
    total_attributed = hub["unlocked_abate"].sum()
    total_abate = activated["abate"].sum()
    hub["HCS_pct"] = hub["unlocked_abate"] / total_attributed * 100.0
    hub = hub.sort_values("HCS_pct", ascending=False)

    ports_gov = ports_dedup.merge(hub, on="port_code", how="left")
    ports_gov["unlocked_abate"] = ports_gov["unlocked_abate"].fillna(0.0)
    ports_gov["n_corridors_served"] = ports_gov["n_corridors_served"].fillna(0).astype(int)
    ports_gov["HCS_pct"] = ports_gov["HCS_pct"].fillna(0.0)

    camp_agg = ports_gov.groupby("camp").agg(
        n_ports=("port_code", "count"),
        n_selected=("is_selected", "sum"),
        selection_rate=("is_selected", "mean"),
        unlocked_abate_Mt=("unlocked_abate", lambda s: s.sum() / 1e6),
    ).reset_index()
    camp_agg["CCS_pct"] = camp_agg["unlocked_abate_Mt"] / (total_attributed / 1e6) * 100.0

    top20 = ports_gov.nlargest(20, "HCS_pct")["HCS_pct"].sum()
    logger.info(f"  Activated abatement: {total_abate/1e6:.2f} Mt; "
                f"top-20 HCS: {top20:.1f}%")
    for _, row in camp_agg.iterrows():
        logger.info(f"    {row['camp']:<6}: {row['n_selected']:>4}/{row['n_ports']:>4} "
                    f"selected, CCS={row['CCS_pct']:5.1f}%")

    ports_gov.to_parquet(POLICY_DIR / f"governance_ports_{vessel_type}.parquet", index=False)
    camp_agg.to_csv(POLICY_DIR / f"camp_control_{vessel_type}.csv", index=False)
    ports_gov[["port_code", "port_name_cn", "country_code", "camp", "is_selected",
               "unlocked_abate", "n_corridors_served", "HCS_pct"]].head(30).to_csv(
        POLICY_DIR / f"hub_control_{vessel_type}.csv", index=False)
    return ports_gov, camp_agg, total_abate, top20


def build_summary(vessel_type: str, gov, ports_gov, camp_agg, p75, rvi_strict,
                  rvi_any, total_abate, top20):
    total_em = gov["E_e"].sum()
    gt = gov.groupby("gov_type").agg(
        n_corridors=("corridor_id", "count"),
        emission_Mt=("E_e", lambda s: s.sum() / 1e6),
    ).reset_index()
    gt["emission_share_pct"] = gt["emission_Mt"] / (total_em / 1e6) * 100.0

    bri_share = gov.loc[gov["is_bri"], "eu_ets_endpoint"].mean() * 100.0
    nonbri_share = gov.loc[~gov["is_bri"], "eu_ets_endpoint"].mean() * 100.0
    top_q = gov["E_e"] >= gov.loc[gov["E_e"] > 0, "E_e"].quantile(0.75)
    h1 = gov.loc[top_q, "gov_type"].eq("Vacuum").mean() * 100.0

    summary = {
        "version": "v2",
        "n_corridors": int(len(gov)),
        "total_emission_Mt": round(total_em / 1e6, 2),
        "p75_Ee_t": round(float(p75), 1),
        "governance_types": gt.set_index("gov_type").to_dict("index"),
        "RVI_any_pct": round(rvi_any, 1),
        "RVI_strict_pct": round(rvi_strict, 1),
        "n_vacuum_highE": int(gov["is_vacuum"].sum()),
        "top25pct_vacuum_share_pct": round(h1, 1),
        "bri_euets_cov_pct": round(bri_share, 1),
        "nonbri_euets_cov_pct": round(nonbri_share, 1),
        "n_fueleu_exposed": int(gov["fueleu_exposed"].sum()),
        "total_activated_abatement_Mt": round(total_abate / 1e6, 2),
        "top20_hub_HCS_pct": round(float(top20), 1),
        "camp_control": camp_agg.set_index("camp")["CCS_pct"].to_dict(),
        "camp_selection": camp_agg.set_index("camp")["selection_rate"].mul(100).round(1).to_dict(),
    }
    with open(POLICY_DIR / f"governance_summary_{vessel_type}.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Saved: governance_summary (v2)")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vessel-type", default=VESSEL_TYPE)
    args = parser.parse_args()
    vt = args.vessel_type
    logger.info("=" * 60)
    logger.info(f"GOVERNANCE CLASSIFICATION — v2 ({vt})")
    logger.info("=" * 60)

    gov, p75, rvi_strict, rvi_any = classify_corridors(vt)
    ports_gov, camp_agg, total_abate, top20 = classify_ports(vt, gov)
    build_summary(vt, gov, ports_gov, camp_agg, p75, rvi_strict, rvi_any,
                  total_abate, top20)
    logger.info("\nDone.")


if __name__ == "__main__":
    main()
