"""
Module 05c-control (v5): M^BRI objective-only control (GLM-R5 A1)
==================================================================
Control experiment isolating the investment-domain restriction of M^BRI:

    max  Σ_{e ∈ E^BRI} W_e·z_e        (BRI-scope objective, as in M^BRI)
    s.t. Σ_{p ∈ P^feas} c_p·y_p ≤ B   (investment domain: ALL feasible ports)

Compared with the domain-restricted M^BRI (05c), the difference in realized
global abatement decomposes the MPC of 57.5% into
  * objective-restriction cost  (this control vs. M^G), and
  * domain-restriction cost     (M^BRI vs. this control).

Reports BOTH activation conventions:
  * solver-z convention (as in 09_fueleu_analysis / 05c): only corridors the
    solver sets z=1 for (the rewarded BRI-scope corridors);
  * maximal-activation convention: every corridor whose endpoints (and
    intermediate-bunkering requirements) are met by the selected ports —
    the global abatement the investment actually unlocks.

Also recomputes the maximal-activation measure for M^G, M^EU, and the
domain-restricted M^BRI from their saved port selections, so all four
models are comparable under one convention.

Outputs (02_数据_output/policy_analysis_v5/):
  bri_view_control_comparison_container.json,
  bri_view_control_selected_ports_container.csv,
  bri_view_maximal_activation_container.json
"""

import sys
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PROCESSED_DIR, POLICY_DIR, MILP_PARAMS, VESSEL_TYPE,
)
from importlib import import_module
_08 = import_module("08_bri_analysis")
BRI_MARITIME_COUNTRIES = _08.BRI_MARITIME_COUNTRIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_activation_context(vessel_type: str):
    """Shared corridor/port context for maximal-activation recomputation."""
    m05 = import_module("05_milp_solving")
    corridors = pd.read_parquet(PROCESSED_DIR / f"corridors_{vessel_type}.parquet")
    ports_dedup = pd.read_parquet(PROCESSED_DIR / f"candidate_ports_{vessel_type}.parquet") \
        .drop_duplicates("port_code")
    mell_raw = pd.read_parquet(PROCESSED_DIR / f"M_ell_{vessel_type}.parquet")
    mell = m05.prune_mell(mell_raw, corridors, ports_dedup,
                          K=MILP_PARAMS["m_ell_prune_k"])
    fuel = MILP_PARAMS["default_fuel"]

    ports_f = ports_dedup[ports_dedup["is_feasible"]].copy()
    port_idx = {p: i for i, p in enumerate(ports_f["port_code"])}

    corr = m05.prepare_corridors(corridors, fuel)
    corr_list = corr[["corridor_id", "origin_port", "dest_port", "E_e", "W_e",
                      "mean_distance", "k_e", "range_violating"]].copy()
    corr_list = corr_list[corr_list["origin_port"].isin(port_idx) &
                          corr_list["dest_port"].isin(port_idx)].reset_index(drop=True)

    bri_corr = set(pd.read_parquet(POLICY_DIR / f"bri_corridors_{vessel_type}.parquet")
                   .loc[lambda d: d["is_bri"], "corridor_id"])
    corr_list["is_bri"] = corr_list["corridor_id"].isin(bri_corr)

    mell_dict = {}
    if len(mell) > 0:
        mell_f = mell[mell["fuel"] == fuel]
        for cid, g in mell_f.groupby("corridor_id"):
            mell_dict[cid] = [p for p in g["intermediate_port"].tolist() if p in port_idx]

    return {
        "corr_list": corr_list, "ports_f": ports_f, "port_idx": port_idx,
        "mell_dict": mell_dict, "fuel": fuel,
    }


def maximal_activation(ctx, selected_ports: set):
    """Corridors unlockable by `selected_ports` under Eq. 14-15 semantics."""
    act = []
    for _, row in ctx["corr_list"].iterrows():
        cid, o, d = row["corridor_id"], row["origin_port"], row["dest_port"]
        if o not in selected_ports or d not in selected_ports:
            continue
        if row["range_violating"]:
            k_e = max(1, int(row["k_e"]))
            m_ports = ctx["mell_dict"].get(cid, [])
            if sum(1 for p in m_ports if p in selected_ports) < k_e:
                continue
        act.append(cid)
    cl = ctx["corr_list"]
    sub = cl[cl["corridor_id"].isin(act)]
    return {
        "n_activated": len(act),
        "global_W_Mt": sub["W_e"].sum() / 1e6,
        "bri_scope_W_Mt": sub.loc[sub["is_bri"], "W_e"].sum() / 1e6,
        "bri_share_pct": sub["is_bri"].mean() * 100 if len(sub) else 0.0,
    }


def run_control(vessel_type: str, budget_fraction: float = 0.20,
                time_limit: int = 900):
    logger.info(f"M^BRI objective-only control: B={budget_fraction:.0%}")
    import pulp
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
    port_idx = {p: i for i, p in enumerate(ports_f["port_code"])}
    base_budget = ports_f["cost_p"].sum() * budget_fraction
    logger.info(f"  Feasible ports: {len(ports_f)} "
                f"(BRI {ports_f['is_bri'].sum()}); B = {base_budget:.1f}")

    corr = m05.prepare_corridors(corridors, fuel)
    corr_list = corr[["corridor_id", "origin_port", "dest_port", "E_e", "W_e",
                      "mean_distance", "k_e", "range_violating"]].copy()
    corr_list = corr_list[corr_list["origin_port"].isin(port_idx) &
                          corr_list["dest_port"].isin(port_idx)].reset_index(drop=True)
    bri_corr = set(pd.read_parquet(POLICY_DIR / f"bri_corridors_{vessel_type}.parquet")
                   .loc[lambda d: d["is_bri"], "corridor_id"])
    corr_list["is_bri"] = corr_list["corridor_id"].isin(bri_corr)
    logger.info(f"  Corridors: {len(corr_list)} (BRI-scope: {corr_list['is_bri'].sum()})")

    mell_dict = {}
    if len(mell) > 0:
        mell_f = mell[mell["fuel"] == fuel]
        for cid, g in mell_f.groupby("corridor_id"):
            mell_dict[cid] = [p for p in g["intermediate_port"].tolist() if p in port_idx]

    prob = pulp.LpProblem("M_BRI_control", pulp.LpMaximize)
    cost_p = ports_f.set_index("port_code")["cost_p"]
    y = {p: pulp.LpVariable(f"y_{p}", cat="Binary") for p in ports_f["port_code"]}
    z = {cid: pulp.LpVariable(f"z_{cid}", cat="Binary")
         for cid in corr_list["corridor_id"]}
    prob += pulp.lpSum(row["W_e"] * z[row["corridor_id"]]
                       for _, row in corr_list.iterrows() if row["is_bri"])
    prob += (pulp.lpSum(cost_p[p] * y[p] for p in ports_f["port_code"])
             <= base_budget, "budget")

    for _, row in corr_list.iterrows():
        cid, o, d = row["corridor_id"], row["origin_port"], row["dest_port"]
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
    t0 = time.time()
    prob.solve(solver)
    logger.info(f"  Solve: {pulp.LpStatus[prob.status]} in {time.time()-t0:.0f}s")
    if prob.status != pulp.LpStatusOptimal:
        logger.warning("  Not optimal — results are the incumbent best")

    sel = [p for p in ports_f["port_code"] if y[p].varValue and y[p].varValue > 0.5]
    act = [cid for cid in corr_list["corridor_id"]
           if z[cid].varValue and z[cid].varValue > 0.5]
    obj_bri = pulp.value(prob.objective)
    obj_global_solver = corr_list.loc[corr_list["corridor_id"].isin(act), "W_e"].sum()

    sel_set = set(sel)
    sel_bri = int(ports_f.set_index("port_code").loc[sel, "is_bri"].sum())
    ctx = {"corr_list": corr_list, "ports_f": ports_f, "port_idx": port_idx,
           "mell_dict": mell_dict}
    ma = maximal_activation(ctx, sel_set)

    bl_json = POLICY_DIR / f"fueleu_comparison_{vessel_type}_viewglobal_b{int(round(budget_fraction*100))}_g0.0_noforce.json"
    bl = json.load(open(bl_json))
    baseline_global = bl["rule_obj_global_Mt"] * 1e6

    bri_json = json.load(open(POLICY_DIR / f"bri_view_comparison_{vessel_type}_b{int(round(budget_fraction*100))}.json"))

    mpc_solver = (baseline_global - obj_global_solver) / baseline_global * 100
    mpc_maximal = (baseline_global - ma["global_W_Mt"] * 1e6) / baseline_global * 100

    out = {
        "model": "M_BRI_control (objective-only, domain free)",
        "budget_fraction": budget_fraction, "fuel": fuel,
        "baseline_global_Mt": round(baseline_global / 1e6, 2),
        "objective_value_bri_Mt": round(obj_bri / 1e6, 2),
        "solver_convention": {
            "global_W_Mt": round(obj_global_solver / 1e6, 2),
            "MPC_pct": round(mpc_solver, 2),
        },
        "maximal_activation": ma,
        "MPC_maximal_pct": round(mpc_maximal, 2),
        "n_ports_selected": len(sel),
        "n_ports_bri": sel_bri,
        "n_ports_non_bri": len(sel) - sel_bri,
        "domain_restricted_M_BRI": {
            "global_W_Mt": bri_json["rule_obj_global_Mt"],
            "MPC_pct": bri_json["MPC_global_pct"],
        },
        "decomposition_solver_convention_pct": {
            "objective_restriction_cost": round(mpc_solver, 2),
            "domain_restriction_cost": round(bri_json["MPC_global_pct"] - mpc_solver, 2),
        },
    }
    logger.info(json.dumps(out, indent=2))

    pd.DataFrame({"port_code": sel}).to_csv(
        POLICY_DIR / f"bri_view_control_selected_ports_{vessel_type}.csv", index=False)
    with open(POLICY_DIR / f"bri_view_control_comparison_{vessel_type}.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


def recompute_maximal_all(vessel_type: str, budget_fraction: float = 0.20):
    """Maximal-activation global abatement for M^G, M^EU, M^BRI, control."""
    ctx = build_activation_context(vessel_type)
    pct = int(round(budget_fraction * 100))
    sources = {
        "M_G": POLICY_DIR / f"fueleu_selected_ports_{vessel_type}_viewglobal_b{pct}_g0.0_noforce.csv",
        "M_EU": POLICY_DIR / f"fueleu_selected_ports_{vessel_type}_vieweu_b{pct}_g0.0_noforce.csv",
        "M_BRI": POLICY_DIR / f"bri_view_selected_ports_{vessel_type}_b{pct}.csv",
        "M_BRI_control": POLICY_DIR / f"bri_view_control_selected_ports_{vessel_type}.csv",
    }
    out = {}
    for name, path in sources.items():
        if not path.exists():
            logger.warning(f"  {name}: {path.name} missing — skipped")
            continue
        sel = set(pd.read_csv(path)["port_code"])
        ma = maximal_activation(ctx, sel)
        out[name] = {
            "n_selected_ports": len(sel),
            "maximal_global_W_Mt": round(ma["global_W_Mt"], 2),
            "maximal_bri_scope_W_Mt": round(ma["bri_scope_W_Mt"], 2),
            "n_activated_corridors": ma["n_activated"],
            "bri_share_activated_pct": round(ma["bri_share_pct"], 1),
        }
        logger.info(f"  {name}: {out[name]}")
    with open(POLICY_DIR / f"bri_view_maximal_activation_{vessel_type}.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    run_control(VESSEL_TYPE)
    recompute_maximal_all(VESSEL_TYPE)
