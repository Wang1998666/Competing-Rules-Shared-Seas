"""
Module 05c (v2): BRI-View MILP (M^BRI) — China Investment Sphere
================================================================
Rule-competition model from the BRI investment perspective (Eq. 24):

    max  Σ_{e ∈ E^BRI} W_e·z_e        (BRI-scope objective)
    s.t. Σ_{p ∈ P^BRI} c_p·y_p ≤ B    (investment domain: BRI ports only)
         (Eq. 14–15, 18 common constraints; fuel-specific W_e and M_ℓ)

Outputs (02_数据_output/policy_analysis_v2/):
  bri_view_selected_ports_{vt}_b{pct}.csv, bri_view_activated_corridors_{vt}_b{pct}.csv,
  bri_view_comparison_{vt}_b{pct}.json
"""

import sys
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PROCESSED_DIR, POLICY_DIR, EMISSION_PARAMS, MILP_PARAMS, VESSEL_TYPE,
)
from importlib import import_module
_08 = import_module("08_bri_analysis")
BRI_MARITIME_COUNTRIES = _08.BRI_MARITIME_COUNTRIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_bri_view_milp(vessel_type: str, budget_fraction: float = 0.20,
                      time_limit: int = 600):
    logger.info(f"Running BRI-view MILP v2 (M^BRI): B={budget_fraction:.0%}")

    from importlib import import_module
    m05 = import_module("05_milp_solving")

    corridors = pd.read_parquet(PROCESSED_DIR / f"corridors_{vessel_type}.parquet")
    ports_dedup = pd.read_parquet(PROCESSED_DIR / f"candidate_ports_{vessel_type}.parquet") \
        .drop_duplicates("port_code")
    mell_raw = pd.read_parquet(PROCESSED_DIR / f"M_ell_{vessel_type}.parquet")
    mell = m05.prune_mell(mell_raw, corridors, ports_dedup,
                          K=MILP_PARAMS["m_ell_prune_k"])

    fuel = MILP_PARAMS["default_fuel"]
    ports_f = ports_dedup[ports_dedup["is_feasible"]].copy()
    ports_f["is_bri"] = ports_f["country_code"].isin(BRI_MARITIME_COUNTRIES)
    bri_ports = ports_f[ports_f["is_bri"]].copy()
    logger.info(f"  BRI feasible ports: {len(bri_ports)} / {len(ports_f)}")

    cost_p = bri_ports.set_index("port_code")["cost_p"]
    port_idx = {p: i for i, p in enumerate(bri_ports["port_code"])}
    base_budget = ports_f["cost_p"].sum() * budget_fraction
    logger.info(f"  BRI cost pool: {cost_p.sum():.1f} M USD/yr; B = {base_budget:.1f}")

    corr = m05.prepare_corridors(corridors, fuel)
    corr_list = corr[["corridor_id", "origin_port", "dest_port", "E_e", "W_e",
                      "mean_distance", "k_e", "range_violating"]].copy()
    corr_list = corr_list[corr_list["origin_port"].isin(port_idx) &
                          corr_list["dest_port"].isin(port_idx)]
    bri_corr = set(pd.read_parquet(POLICY_DIR / f"bri_corridors_{vessel_type}.parquet")
                   .loc[lambda d: d["is_bri"], "corridor_id"])
    corr_list["is_bri"] = corr_list["corridor_id"].isin(bri_corr)
    logger.info(f"  Corridors: {len(corr_list)} (BRI: {corr_list['is_bri'].sum()})")

    mell_dict = {}
    if len(mell) > 0:
        mell_f = mell[mell["fuel"] == fuel]
        for cid, g in mell_f.groupby("corridor_id"):
            mell_dict[cid] = [p for p in g["intermediate_port"].tolist() if p in port_idx]

    import pulp
    prob = pulp.LpProblem("M_BRI_v2", pulp.LpMaximize)

    y = {p: pulp.LpVariable(f"y_{p}", cat="Binary") for p in bri_ports["port_code"]}
    z = {cid: pulp.LpVariable(f"z_{cid}", cat="Binary")
         for cid in corr_list["corridor_id"]}
    prob += pulp.lpSum(row["W_e"] * z[row["corridor_id"]]
                       for _, row in corr_list.iterrows() if row["is_bri"])
    prob += (pulp.lpSum(cost_p[p] * y[p] for p in bri_ports["port_code"])
             <= base_budget, "budget")

    for _, row in corr_list.iterrows():
        cid, o, d = row["corridor_id"], row["origin_port"], row["dest_port"]
        if o not in port_idx or d not in port_idx:
            prob += (z[cid] == 0, f"noep_{cid}")
            continue
        prob += (z[cid] <= y[o], f"ep1_{cid}")
        prob += (z[cid] <= y[d], f"ep2_{cid}")
        if row["range_violating"]:
            k_e = max(1, int(row["k_e"]))
            m_ports = mell_dict.get(cid, [])
            if len(m_ports) >= k_e:
                prob += (z[cid] * k_e <= pulp.lpSum(y[m] for m in m_ports),
                         f"int_{cid}")
            else:
                prob += (z[cid] == 0, f"nointer_{cid}")

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit,
                               gapRel=MILP_PARAMS["mip_gap"])
    prob.solve(solver)
    if prob.status != pulp.LpStatusOptimal:
        logger.warning(f"  MILP status {pulp.LpStatus[prob.status]} — skipping")
        return None

    sel_ports = [p for p in bri_ports["port_code"]
                 if y[p].varValue and y[p].varValue > 0.5]
    act_corr = [cid for cid in corr_list["corridor_id"]
                if z[cid].varValue and z[cid].varValue > 0.5]
    obj_bri = pulp.value(prob.objective)
    obj_global = corr_list.loc[corr_list["corridor_id"].isin(act_corr), "W_e"].sum()

    # Baseline: same-budget M^G (09 no-force)
    bl_json = POLICY_DIR / f"fueleu_comparison_{vessel_type}_viewglobal_b{int(round(budget_fraction*100))}_g0.0_noforce.json"
    if bl_json.exists():
        bl = json.load(open(bl_json))
        baseline_global = bl["rule_obj_global_Mt"] * 1e6
        baseline_label = "M^G no-force"
    else:
        baseline_global = obj_global
        baseline_label = "self-baseline (M^G not found)"
    mpc = (baseline_global - obj_global) / baseline_global * 100.0 if baseline_global > 0 else float("nan")

    comparison = {
        "version": "v2", "model": "M_BRI", "budget_fraction": budget_fraction,
        "fuel": fuel, "baseline": baseline_label,
        "baseline_global_Mt": round(baseline_global / 1e6, 2),
        "bri_obj_Mt": round(obj_bri / 1e6, 2),
        "rule_obj_global_Mt": round(obj_global / 1e6, 2),
        "MPC_global_pct": round(mpc, 2) if mpc == mpc else None,
        "n_ports_selected": len(sel_ports),
        "n_corridors_activated": len(act_corr),
        "bri_share_activated_pct": round(corr_list.loc[corr_list["corridor_id"].isin(act_corr), "is_bri"].mean() * 100, 1),
    }
    logger.info(f"  M^BRI: obj={obj_bri/1e6:.2f} Mt, global={obj_global/1e6:.2f} Mt "
                f"(MPC={comparison['MPC_global_pct']}%), ports={len(sel_ports)}")

    pct = int(round(budget_fraction * 100))
    pd.DataFrame({"port_code": sel_ports}).to_csv(
        POLICY_DIR / f"bri_view_selected_ports_{vessel_type}_b{pct}.csv", index=False)
    pd.DataFrame({"corridor_id": act_corr}).to_csv(
        POLICY_DIR / f"bri_view_activated_corridors_{vessel_type}_b{pct}.csv", index=False)
    with open(POLICY_DIR / f"bri_view_comparison_{vessel_type}_b{pct}.json", "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"  Saved: bri_view_*_b{pct} (v2)")
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vessel-type", default=VESSEL_TYPE)
    parser.add_argument("--budget-fraction", type=float, default=0.20)
    args = parser.parse_args()
    run_bri_view_milp(args.vessel_type, budget_fraction=args.budget_fraction)
