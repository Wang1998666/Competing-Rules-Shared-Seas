"""
Module 05d (v2): Multi-Objective MILP (M^MOO) — ε-Constraint Frontiers
======================================================================
Two ε-constraint programs (Eq. 26–27):

  * obj2='vac'       : max efficiency s.t. vacuum coverage ≥ ε·U²_max
  * obj2='bri_share' : max efficiency s.t. BRI abatement share ≥ s
                       (real efficiency–equity frontier)

v2: fuel-specific W_e via prepare_corridors; feasibility mask; tighter
MIP gap (5e-4) and longer time limit for frontier quality.

Outputs (02_数据_output/policy_analysis_v2/): moo_pareto_*_{vt}.csv,
moo_summary_*_{vt}.json
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EPS_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
BRI_SHARE_GRID = [0.80, 0.82, 0.85, 0.88, 0.90]


def build_shared_inputs(vessel_type: str):
    from importlib import import_module
    m05 = import_module("05_milp_solving")

    corridors = pd.read_parquet(PROCESSED_DIR / f"corridors_{vessel_type}.parquet")
    gov = pd.read_parquet(POLICY_DIR / f"governance_corridors_{vessel_type}.parquet")
    bri = pd.read_parquet(POLICY_DIR / f"bri_corridors_{vessel_type}.parquet")
    ports_dedup = pd.read_parquet(PROCESSED_DIR / f"candidate_ports_{vessel_type}.parquet") \
        .drop_duplicates("port_code")
    mell_raw = pd.read_parquet(PROCESSED_DIR / f"M_ell_{vessel_type}.parquet")
    mell = m05.prune_mell(mell_raw, corridors, ports_dedup,
                          K=MILP_PARAMS["m_ell_prune_k"])

    fuel = MILP_PARAMS["default_fuel"]
    corr = m05.prepare_corridors(corridors, fuel)
    corr = corr.merge(gov[["corridor_id", "gov_type", "is_vacuum"]], on="corridor_id",
                      how="left")
    corr = corr.merge(bri[["corridor_id", "is_bri"]], on="corridor_id", how="left")
    corr["is_vacuum"] = corr["is_vacuum"].fillna(False)
    corr["is_bri"] = corr["is_bri"].fillna(False)

    ports_f = ports_dedup[ports_dedup["is_feasible"]].copy()
    cost_p = ports_f.set_index("port_code")["cost_p"]
    port_idx = {p: i for i, p in enumerate(ports_f["port_code"])}
    corr = corr[corr["origin_port"].isin(port_idx) & corr["dest_port"].isin(port_idx)]

    mell_dict = {}
    if len(mell) > 0:
        mell_f = mell[mell["fuel"] == fuel]
        for cid, g in mell_f.groupby("corridor_id"):
            mell_dict[cid] = [p for p in g["intermediate_port"].tolist() if p in port_idx]

    return corr, ports_f, cost_p, port_idx, mell_dict


def build_model(corr, cost_p, port_idx, mell_dict, base_budget):
    import pulp
    prob = pulp.LpProblem("M_MOO_v2", pulp.LpMaximize)

    port_list = list(cost_p.index)
    y = {p: pulp.LpVariable(f"y_{p}", cat="Binary") for p in port_list}
    z = {cid: pulp.LpVariable(f"z_{cid}", cat="Binary") for cid in corr["corridor_id"]}
    prob += (pulp.lpSum(cost_p[p] * y[p] for p in port_list) <= base_budget,
             "budget")
    for _, row in corr.iterrows():
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
    return prob, y, z


def run_frontier(vessel_type: str, budget_fraction: float = 0.20, obj2: str = "vac"):
    import pulp
    logger.info(f"Running MOO frontier v2: B={budget_fraction:.0%}, obj2={obj2}")

    corr, ports, cost_p, port_idx, mell_dict = build_shared_inputs(vessel_type)
    base_budget = cost_p.sum() * budget_fraction
    total_we = corr["W_e"].sum()
    vac_we = corr.loc[corr["is_vacuum"], "W_e"].sum()
    logger.info(f"  Corridors={len(corr)}, total W_e={total_we/1e6:.2f} Mt, "
                f"vacuum W_e={vac_we/1e6:.2f} Mt, B={base_budget:.1f} M USD/yr")

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=900, gapRel=0.0005)

    # Reference: pure efficiency
    m_ref, y_ref, z_ref = build_model(corr, cost_p, port_idx, mell_dict, base_budget)
    m_ref += pulp.lpSum(row["W_e"] * z_ref[row["corridor_id"]]
                        for _, row in corr.iterrows())
    m_ref.solve(solver)
    if m_ref.status != pulp.LpStatusOptimal:
        logger.warning("  Reference infeasible")
        return None
    ref_act = [cid for cid in corr["corridor_id"]
               if z_ref[cid].varValue and z_ref[cid].varValue > 0.5]
    ref_eff = corr.loc[corr["corridor_id"].isin(ref_act), "W_e"].sum()
    ref_vac = corr.loc[corr["corridor_id"].isin(ref_act) & corr["is_vacuum"], "W_e"].sum()
    ref_bri = corr.loc[corr["corridor_id"].isin(ref_act) & corr["is_bri"], "W_e"].sum() / max(ref_eff, 1e-9)
    logger.info(f"  Reference: eff={ref_eff/1e6:.2f} Mt, vac={ref_vac/1e6:.2f} Mt, "
                f"BRI share={ref_bri*100:.1f}%")

    if obj2 == "vac":
        u2max = ref_vac if ref_vac > 0 else 1e-6
        grid = [(f"eps={eps:.2f}", eps * u2max) for eps in EPS_GRID]
    else:
        base_share = ref_bri
        grid = [(f"share={s:.2f}", s) for s in BRI_SHARE_GRID if s > base_share + 0.005]
        grid = [("share=free", base_share)] + grid
        if not grid[1:]:
            logger.warning("  Frontier degenerate (no shares above free value)")

    rows = []
    for label, bound in grid:
        m, y, z = build_model(corr, cost_p, port_idx, mell_dict, base_budget)
        m += pulp.lpSum(row["W_e"] * z[row["corridor_id"]]
                        for _, row in corr.iterrows())
        if obj2 == "vac":
            m += (pulp.lpSum(row["W_e"] * z[row["corridor_id"]]
                             for _, row in corr.iterrows() if row["is_vacuum"])
                  >= bound, "vac_eps")
        else:
            m += (pulp.lpSum(row["W_e"] * z[row["corridor_id"]]
                             for _, row in corr.iterrows() if row["is_bri"])
                  - bound * pulp.lpSum(row["W_e"] * z[row["corridor_id"]]
                                       for _, row in corr.iterrows())
                  >= 0, "bri_share")
        m.solve(solver)
        if m.status != pulp.LpStatusOptimal:
            logger.warning(f"  {label}: infeasible — skipping")
            continue
        act = [cid for cid in corr["corridor_id"]
               if z[cid].varValue and z[cid].varValue > 0.5]
        f1 = corr.loc[corr["corridor_id"].isin(act), "W_e"].sum()
        f2_vac = corr.loc[corr["corridor_id"].isin(act) & corr["is_vacuum"], "W_e"].sum()
        bri_share = corr.loc[corr["corridor_id"].isin(act) & corr["is_bri"], "W_e"].sum() / max(f1, 1e-9)
        n_sel = sum(1 for p in cost_p.index
                    if y[p].varValue and y[p].varValue > 0.5)
        rows.append({"label": label, "eff_obj_Mt": round(f1 / 1e6, 2),
                     "vac_obj_Mt": round(f2_vac / 1e6, 2),
                     "vac_share_pct": round(f2_vac / max(f1, 1e-9) * 100, 1),
                     "n_ports": n_sel, "n_corr": len(act),
                     "bri_share_pct": round(bri_share * 100, 1),
                     "status": pulp.LpStatus[m.status]})
        logger.info(f"  {label}: eff={f1/1e6:.2f} Mt, vac={f2_vac/1e6:.2f} Mt, "
                    f"bri={bri_share*100:.1f}%")

    out = pd.DataFrame(rows)
    out.to_csv(POLICY_DIR / f"moo_pareto_{vessel_type}_{obj2}.csv", index=False)
    summary = {"version": "v2", "budget_fraction": budget_fraction, "obj2": obj2,
               "u2max_Mt": round(u2max / 1e6, 2) if obj2 == "vac" else None,
               "ref_eff_Mt": round(ref_eff / 1e6, 2),
               "ref_bri_share_pct": round(ref_bri * 100, 1),
               "n_points": len(rows)}
    with open(POLICY_DIR / f"moo_summary_{vessel_type}_{obj2}.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Saved: moo_pareto_{obj2} (v2)")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vessel-type", default=VESSEL_TYPE)
    parser.add_argument("--budget-fraction", type=float, default=0.20)
    parser.add_argument("--obj2", default="vac", choices=["vac", "bri_share"])
    args = parser.parse_args()
    run_frontier(args.vessel_type, budget_fraction=args.budget_fraction, obj2=args.obj2)
