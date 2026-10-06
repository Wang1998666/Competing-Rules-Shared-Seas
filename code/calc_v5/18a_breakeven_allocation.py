"""
Module 18a (v4): Break-Even Price — Dual Accounting Tracks
==========================================================
Addresses review point A: the standalone corridor-level infrastructure
allocation double-counts hub-port costs (each corridor books 0.5*(c_o+c_d)
+ k_e*mean_intermediate), so the corridor sum (114.8 B USD/yr) is ~7.5x
the true full-deployment cost (15.4 B USD/yr over 944 feasible ports).

This module adds a utilization-based allocation track in which every
feasible port's annualized cost is split across the corridors that use it
(as endpoint or as candidate intermediate bunkering port) in proportion to
each corridor's abatement dE_e, so that corridor-level allocations sum
exactly to the true total cost:

    C^alloc_e = sum_{p in ports(e)} c_p * dE_e / sum_{e' in U(p)} dE_{e'}

Three tracks are reported per corridor:
  * standalone (original, upper bound; hub costs multiply booked)
  * allocated  (utilization-based, budget-consistent accounting)
  * fuel-only  (no infrastructure component at all; lower bound)

Outputs (02_数据_output/policy_analysis_v4/):
  breakeven_dualtrack_container.csv
  breakeven_dualtrack_summary_container.json
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import PROCESSED_DIR, POLICY_DIR, VESSEL_TYPE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FUEL = "green_methanol"


def main(vessel_type: str):
    corridors = pd.read_parquet(PROCESSED_DIR / f"corridors_{vessel_type}.parquet")
    ports = pd.read_parquet(PROCESSED_DIR / f"candidate_ports_{vessel_type}.parquet")
    mell = pd.read_parquet(PROCESSED_DIR / f"M_ell_{vessel_type}.parquet")
    bp = pd.read_csv(POLICY_DIR / f"breakeven_prices_{vessel_type}.csv")

    feas = ports[ports["is_feasible"]].copy()
    cost_p = feas.set_index("port_code")["cost_p"]  # M USD/yr, feasible ports only
    true_total = float(cost_p.sum())
    logger.info(f"Feasible ports: {len(feas)}; true full cost pool: {true_total:,.0f} M USD/yr")

    # corridor abatement weights (methanol reference)
    dE = bp.set_index("corridor_id")["dE_e"]
    w = dE.clip(lower=0.0)

    # --- build port->corridor usage links -------------------------------
    links = []  # (port, corridor_id)
    for _, row in bp.iterrows():
        links.append((row["origin_port"], row["corridor_id"]))
        links.append((row["dest_port"], row["corridor_id"]))
    mell_f = mell[mell["fuel"] == FUEL]
    links += list(zip(mell_f["intermediate_port"], mell_f["corridor_id"]))
    links_df = pd.DataFrame(links, columns=["port_code", "corridor_id"]).drop_duplicates()
    links_df = links_df[links_df["port_code"].isin(cost_p.index)]
    links_df["dE"] = links_df["corridor_id"].map(w).fillna(0.0)

    # per-port normalization: share of port cost allocated to each user corridor
    links_df["port_total_dE"] = links_df.groupby("port_code")["dE"].transform("sum")
    pos = links_df[links_df["port_total_dE"] > 0].copy()
    pos["cost_share_MUSD"] = (cost_p.reindex(pos["port_code"]).to_numpy()
                              * pos["dE"].to_numpy() / pos["port_total_dE"].to_numpy())
    alloc = pos.groupby("corridor_id")["cost_share_MUSD"].sum()
    alloc_total = float(alloc.sum())
    unallocated = true_total - alloc_total
    n_unalloc_ports = int((links_df.groupby("port_code")["port_total_dE"].max() == 0).sum())
    logger.info(f"Allocated total: {alloc_total:,.0f} M USD/yr "
                f"(unallocated residual: {unallocated:,.0f} M from "
                f"{n_unalloc_ports} ports used only by zero-abatement corridors)")

    # --- dual-track breakeven prices -------------------------------------
    out = bp.copy()
    out["C_infra_alloc_MUSD"] = out["corridor_id"].map(alloc).fillna(0.0)
    out["C_infra_alloc_usd"] = out["C_infra_alloc_MUSD"] * 1e6
    with np.errstate(divide="ignore", invalid="ignore"):
        out["p_star_alloc_usd"] = ((out["dC_fuel_usd"] + out["C_infra_alloc_usd"])
                                   / out["dE_e"].clip(lower=1e-9))
    out["p_star_alloc_usd"] = out["p_star_alloc_usd"].where(out["dE_e"] > 0, np.nan)
    out["p_fuel_only_usd"] = (out["dC_fuel_usd"] / out["dE_e"].clip(lower=1e-9)).where(
        out["dE_e"] > 0, np.nan)

    def stats(s):
        v = s.dropna()
        return {"n": int(len(v)), "median": float(v.median()),
                "p25": float(v.quantile(0.25)), "p75": float(v.quantile(0.75)),
                "share_le_380": float((v <= 380).mean()),
                "share_le_1000": float((v <= 1000).mean()),
                "share_le_5000": float((v <= 5000).mean())}

    gov = pd.read_parquet(POLICY_DIR / f"governance_corridors_{vessel_type}.parquet")
    out = out.merge(gov[["corridor_id", "gov_type"]], on="corridor_id", how="left")

    summary = {
        "fuel": FUEL,
        "true_full_cost_MUSD": round(true_total, 1),
        "standalone_sum_MUSD": round(float(out["C_infra_usd"].sum() / 1e6), 1),
        "allocated_sum_MUSD": round(alloc_total, 1),
        "unallocated_residual_MUSD": round(unallocated, 1),
        "n_unallocated_ports": n_unalloc_ports,
        "double_counting_factor_standalone": round(
            float(out["C_infra_usd"].sum() / 1e6 / true_total), 2),
        "dC_fuel_total_MUSD": round(float(out["dC_fuel_usd"].sum() / 1e6), 1),
        "tracks": {
            "standalone": stats(out["p_star_usd"]),
            "allocated": stats(out["p_star_alloc_usd"]),
            "fuel_only": stats(out["p_fuel_only_usd"]),
        },
        "by_gov_type": {
            gt: {"standalone_median": float(sub["p_star_usd"].median()),
                 "allocated_median": float(sub["p_star_alloc_usd"].median()),
                 "fuel_only_median": float(sub["p_fuel_only_usd"].median())}
            for gt, sub in out.groupby("gov_type")
        },
    }

    keep = ["corridor_id", "origin_port", "dest_port", "E_e", "dE_e",
            "dC_fuel_usd", "C_infra_usd", "C_infra_alloc_usd",
            "p_star_usd", "p_star_alloc_usd", "p_fuel_only_usd", "gov_type"]
    out[keep].to_csv(POLICY_DIR / f"breakeven_dualtrack_{vessel_type}.csv", index=False)
    with open(POLICY_DIR / f"breakeven_dualtrack_summary_{vessel_type}.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Track medians (USD/tCO2): standalone "
                f"{summary['tracks']['standalone']['median']:.0f}, allocated "
                f"{summary['tracks']['allocated']['median']:.0f}, fuel-only "
                f"{summary['tracks']['fuel_only']['median']:.0f}")
    logger.info("Share of corridors viable at 380 USD/tCO2: standalone "
                f"{summary['tracks']['standalone']['share_le_380']:.3f}, allocated "
                f"{summary['tracks']['allocated']['share_le_380']:.3f}, fuel-only "
                f"{summary['tracks']['fuel_only']['share_le_380']:.3f}")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vessel-type", default=VESSEL_TYPE,
                        choices=["container", "tanker"])
    main(parser.parse_args().vessel_type)
