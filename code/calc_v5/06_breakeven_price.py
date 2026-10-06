"""
Module 06 (v2): Break-Even Carbon Prices (NEW — Eq. 28)
========================================================
For each corridor e:

    p*_e = (ΔC^fuel_e + CRF·C^infra_e) / ΔE_e            (Eq. 28)

  * ΔC^fuel_e: annual fuel-cost increment. The corridor's annual main-engine
    energy (kWh) is recovered from its leg emissions:
        E_kWh_e = Σ_voyages E_odv / EF̄_e · 1e6
    and converted with fuel prices and lower heating values:
        ΔC^fuel_e = E_kWh_e · 3.6 · (p_alt/LHV_alt − p_conv/LHV_conv) / 1000  [USD]
  * C^infra_e: annualized infrastructure cost allocated to the corridor —
    the annualized costs of its endpoint ports plus k_e intermediate ports,
    discounted by a half-sharing factor (both endpoints jointly unlock the
    corridor; Eq. 17 costs).
  * ΔE_e = E_e·η_e·α_e (methanol reference scenario).

Outputs (02_数据_output/policy_analysis_v2/):
  breakeven_prices_{vt}.csv (corridor level)
  breakeven_summary_{vt}.json (per-camp distributions)
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
    PROCESSED_DIR, RESULTS_DIR, POLICY_DIR, EMISSION_PARAMS, COST_PARAMS,
    VESSEL_TYPE, get_data_paths,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def crf(r_d: float, t_h: int) -> float:
    return r_d * (1 + r_d) ** t_h / ((1 + r_d) ** t_h - 1)


def compute_breakeven(vessel_type: str, fuel: str = "green_methanol"):
    logger.info(f"Break-even carbon prices v2: {vessel_type} ({fuel})")

    corridors = pd.read_parquet(PROCESSED_DIR / f"corridors_{vessel_type}.parquet")
    emissions = pd.read_parquet(PROCESSED_DIR / f"emissions_{vessel_type}.parquet")
    ports = pd.read_parquet(PROCESSED_DIR / f"candidate_ports_{vessel_type}.parquet")
    mell = pd.read_parquet(PROCESSED_DIR / f"M_ell_{vessel_type}.parquet")

    prices = EMISSION_PARAMS["fuel_prices_usd_t"]
    lhv = EMISSION_PARAMS["fuel_lhv_mj_kg"]
    alt, conv = fuel, "HFO"

    # Annual main-engine energy per corridor (kWh/yr)
    em = emissions.dropna(subset=["origin_port", "dest_port"]).copy()
    em["origin_port"] = em["origin_port"].astype(str)
    em["dest_port"] = em["dest_port"].astype(str)
    em["p1"] = np.minimum(em["origin_port"], em["dest_port"])
    em["p2"] = np.maximum(em["origin_port"], em["dest_port"])
    em["energy_kwh"] = em["emission_tco2e"] / em["ef_wtw"] * 1e6
    energy_e = em.groupby(["p1", "p2"])["energy_kwh"].sum().reset_index() \
                 .rename(columns={"p1": "origin_port", "p2": "dest_port",
                                  "energy_kwh": "energy_kwh_e"})
    corridors = corridors.merge(energy_e, on=["origin_port", "dest_port"], how="left")
    corridors["energy_kwh_e"] = corridors["energy_kwh_e"].fillna(0.0)

    # Fuel cost increment per kWh (USD/kWh)
    d_c_kwh = (3.6 / 1000) * (prices[alt] / lhv[alt] - prices[conv] / lhv[conv])
    corridors["dC_fuel_usd"] = corridors["energy_kwh_e"] * d_c_kwh

    # Infrastructure cost allocation (Eq. 28): endpoints + k_e intermediates,
    # half-shared
    cost_p = ports.set_index("port_code")["cost_p"]
    crf_val = crf(COST_PARAMS["discount_rate"], COST_PARAMS["planning_horizon_years"])
    share = EMISSION_PARAMS["infra_cost_sharing"]

    kcol = f"k_e_{fuel}"
    mell_fuel = mell[mell["fuel"] == fuel] if len(mell) else mell
    inter_cost = {}
    if len(mell_fuel):
        for cid, g in mell_fuel.groupby("corridor_id"):
            cs = cost_p.reindex(g["intermediate_port"].tolist()).fillna(0.0)
            inter_cost[cid] = cs.sum() / max(len(cs), 1)  # mean intermediate cost

    c_infra = []
    for _, row in corridors.iterrows():
        co = cost_p.get(row["origin_port"], 0.0)
        cd = cost_p.get(row["dest_port"], 0.0)
        k_e = max(0, int(row.get(kcol, 0)))
        cm = inter_cost.get(row["corridor_id"], 0.0)
        c_infra.append(share * (co + cd + k_e * cm))
    # cost_p is in M USD/yr → convert to USD/yr for Eq. 28
    corridors["C_infra_usd"] = np.array(c_infra) * 1e6

    # ΔE_e (Eq. 28 denominator)
    eta_col = f"eta_{fuel}"
    corridors["dE_e"] = corridors["E_e"] * corridors[eta_col] * corridors["alpha_e"]

    with np.errstate(divide="ignore", invalid="ignore"):
        p_star = (corridors["dC_fuel_usd"] + corridors["C_infra_usd"]) / \
                 corridors["dE_e"].clip(lower=1e-9)
    corridors["p_star_usd"] = p_star.where(corridors["dE_e"] > 0, np.nan)

    out = corridors[["corridor_id", "origin_port", "dest_port", "E_e", "dE_e",
                     "energy_kwh_e", "dC_fuel_usd", "C_infra_usd",
                     "p_star_usd"]].copy()
    out.to_csv(POLICY_DIR / f"breakeven_prices_{vessel_type}.csv", index=False)

    valid = out["p_star_usd"].dropna()
    logger.info(f"  Corridors with valid p*: {len(valid):,}")
    logger.info(f"  Median p*: {valid.median():.1f} USD/tCO2; "
                f"mean: {valid.mean():.1f}; "
                f"p25: {valid.quantile(0.25):.1f}; p75: {valid.quantile(0.75):.1f}")

    # Per-camp summary (if governance labels exist)
    summary = {"version": "v2", "fuel": fuel,
               "n_corridors": int(len(valid)),
               "median_p_star": float(valid.median()),
               "mean_p_star": float(valid.mean()),
               "p25_p_star": float(valid.quantile(0.25)),
               "p75_p_star": float(valid.quantile(0.75)),
               "dC_fuel_total_MUSD": float(out["dC_fuel_usd"].sum() / 1e6),
               "C_infra_total_MUSD": float(out["C_infra_usd"].sum() / 1e6)}
    gov_file = POLICY_DIR / f"governance_corridors_{vessel_type}.parquet"
    if gov_file.exists():
        gov = pd.read_parquet(gov_file)[["corridor_id", "gov_type"]]
        merged = out.merge(gov, on="corridor_id")
        camp_stats = {}
        for gt, sub in merged.groupby("gov_type"):
            pv = sub["p_star_usd"].dropna()
            camp_stats[gt] = {"n": int(len(pv)),
                              "median_p_star": float(pv.median()),
                              "p25": float(pv.quantile(0.25)),
                              "p75": float(pv.quantile(0.75))}
        summary["by_gov_type"] = camp_stats
    with open(POLICY_DIR / f"breakeven_summary_{vessel_type}.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Saved: breakeven_prices / breakeven_summary (v2)")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vessel-type", default=VESSEL_TYPE)
    parser.add_argument("--fuel", default="green_methanol")
    args = parser.parse_args()
    compute_breakeven(args.vessel_type, args.fuel)
