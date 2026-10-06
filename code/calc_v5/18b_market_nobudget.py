"""
Module 18b (v4): Market-Mechanism Experiment — Capital Constraint & Cost Track
==============================================================================
Addresses review point B: Eq. (market) retains the 20% budget constraint, so
the "market converges to planning only at 5,000 USD/tCO2" finding partly
reflects the capital cap (at c=5,000 the market plan spends 3,049 of the
3,070 M budget — 99.3% saturation).

This module re-runs the net-profit MILP under a 2x2 design:
  * capital:   constrained (same 20% pool as M^G) vs unconstrained (pure
               price-coordinated private investment)
  * cost track: standalone p* (original, upper bound) vs allocated p*
               (utilization-based, budget-consistent; Module 18a)

Prices: c in {380, 1000, 2000, 5000} USD/tCO2 (plus 10000 for the
allocated track to locate the convergence price).

Outputs (02_数据_output/policy_analysis_v4/):
  market_design_summary_container.json
  market_unconstrained_{track}_{c}_{vt}.csv (activated corridors)
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import PROCESSED_DIR, POLICY_DIR, MILP_PARAMS, VESSEL_TYPE
import importlib
m05 = importlib.import_module("05_milp_solving")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FUEL = "green_methanol"
CARBON_PRICES = [380, 1000, 2000, 5000, 10000]


def solve_market(corridors, ports, M_ell_df, budget, carbon_price, bp_map,
                 p_col="p_star_usd"):
    """Net-profit MILP; budget=None removes the capital constraint."""
    corr = m05.prepare_corridors(corridors, FUEL)
    ports_f = ports[ports["is_feasible"]].copy()
    port_list = ports_f["port_code"].tolist()
    port_cost = ports_f.set_index("port_code")["cost_p"].to_dict()
    port_idx = {p: i for i, p in enumerate(port_list)}

    mell_fuel = M_ell_df[M_ell_df["fuel"] == FUEL] if len(M_ell_df) else M_ell_df
    corr_intermediates = {}
    if len(mell_fuel) > 0:
        for cid, g in mell_fuel.groupby("corridor_id"):
            corr_intermediates[cid] = [p for p in g["intermediate_port"].tolist()
                                       if p in port_idx]

    profit = {}
    for _, row in corr.iterrows():
        cid = row["corridor_id"]
        b = bp_map.get(cid)
        if b is None or b[p_col] is None or np.isnan(b[p_col]):
            profit[cid] = -1e18
        else:
            profit[cid] = (carbon_price - b[p_col]) * b["dE_e"]

    import pulp
    prob = pulp.LpProblem("Market_MILP", pulp.LpMaximize)
    y = {p: pulp.LpVariable(f"y_{p}", cat="Binary") for p in port_list}
    z = {cid: pulp.LpVariable(f"z_{cid}", cat="Binary") for cid in corr["corridor_id"]}

    # epsilon-cost on ports: without it, y is free when unconstrained (solver
    # returns arbitrary y); the 1e-6 M USD (=1 USD) penalty is ~1e-11 of the
    # objective and only prunes unneeded ports
    prob += (pulp.lpSum(profit[cid] * z[cid] for cid in z)
             - 1e-6 * pulp.lpSum(port_cost[p] * y[p] for p in port_list), "Net_Profit")
    if budget is not None:
        prob += (pulp.lpSum(port_cost[p] * y[p] for p in port_list) <= budget, "Budget")

    for _, row in corr.iterrows():
        cid = row["corridor_id"]
        o, d = row["origin_port"], row["dest_port"]
        if o not in port_idx or d not in port_idx:
            prob += z[cid] == 0, f"NoPort_{cid}"
            continue
        prob += z[cid] <= y[o], f"zYo_{cid}"
        prob += z[cid] <= y[d], f"zYd_{cid}"
        if row["range_violating"]:
            k_e = max(1, int(row["k_e"]))
            inter = corr_intermediates.get(cid, [])
            if len(inter) >= k_e:
                prob += (z[cid] * k_e <= pulp.lpSum(y[p] for p in inter), f"zInter_{cid}")
            else:
                prob += z[cid] == 0, f"NoInter_{cid}"

    try:
        solver = pulp.GUROBI(msg=0, timeLimit=MILP_PARAMS["time_limit_s"],
                             MIPGap=MILP_PARAMS["mip_gap"])
        prob.solve(solver)
    except Exception:
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=MILP_PARAMS["time_limit_s"],
                                   gapRel=MILP_PARAMS["mip_gap"])
        prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    obj_val = pulp.value(prob.objective) if prob.status == 1 else 0.0
    selected = [p for p in port_list if y[p].varValue and y[p].varValue > 0.5]
    activated = [cid for cid in z if z[cid].varValue and z[cid].varValue > 0.5]
    W = corr.set_index("corridor_id")["W_e"].to_dict()
    capex = sum(port_cost[p] for p in selected)

    return {"status": status, "net_profit_usd": obj_val,
            "abatement_tco2e": sum(W.get(cid, 0.0) for cid in activated),
            "n_ports": len(selected), "n_corridors": len(activated),
            "capex_MUSD": capex, "activated_corridors": activated}


def main(vessel_type: str):
    corridors = pd.read_parquet(PROCESSED_DIR / f"corridors_{vessel_type}.parquet")
    ports = pd.read_parquet(PROCESSED_DIR / f"candidate_ports_{vessel_type}.parquet")
    M_ell_df = pd.read_parquet(PROCESSED_DIR / f"M_ell_{vessel_type}.parquet")
    bp_std = pd.read_csv(POLICY_DIR / f"breakeven_prices_{vessel_type}.csv")
    bp_alloc = pd.read_csv(POLICY_DIR / f"breakeven_dualtrack_{vessel_type}.csv")
    bp_map_std = bp_std.set_index("corridor_id").to_dict("index")
    bp_map_alloc = bp_alloc.set_index("corridor_id").to_dict("index")

    M_ell_pruned = m05.prune_mell(M_ell_df, corridors, ports,
                                  K=MILP_PARAMS["m_ell_prune_k"])
    total_cost = ports.loc[ports["is_feasible"], "cost_p"].sum()
    budget = total_cost * 0.20
    logger.info(f"Budget (20%): {budget:.1f} M USD/yr; prices: {CARBON_PRICES}")

    results = []
    runs = [
        ("standalone", "constrained", budget, bp_map_std, "p_star_usd"),
        ("standalone", "unconstrained", None, bp_map_std, "p_star_usd"),
        ("allocated", "constrained", budget, bp_map_alloc, "p_star_alloc_usd"),
        ("allocated", "unconstrained", None, bp_map_alloc, "p_star_alloc_usd"),
    ]
    for track, cap, bud, bp_map, p_col in runs:
        for c in CARBON_PRICES:
            t0 = time.time()
            r = solve_market(corridors, ports, M_ell_pruned, bud, c, bp_map, p_col)
            row = {"track": track, "capital": cap, "carbon_price": c,
                   "abatement_Mt": r["abatement_tco2e"] / 1e6,
                   "n_ports": r["n_ports"], "n_corridors": r["n_corridors"],
                   "capex_MUSD": r["capex_MUSD"], "status": r["status"]}
            results.append(row)
            logger.info(f"[{track}/{cap}] c={c}: {row['abatement_Mt']:.2f} Mt, "
                        f"{r['n_ports']} ports, capex {r['capex_MUSD']:.0f} M, "
                        f"{r['status']} ({time.time()-t0:.0f}s)")
            if r["activated_corridors"] and cap == "unconstrained":
                act = corridors[corridors["corridor_id"].isin(r["activated_corridors"])]
                act.to_csv(POLICY_DIR / f"market_unconstrained_{track}_{c}_{vessel_type}.csv",
                           index=False)

    df = pd.DataFrame(results)
    df.to_csv(POLICY_DIR / f"market_design_summary_{vessel_type}.csv", index=False)
    with open(POLICY_DIR / f"market_design_summary_{vessel_type}.json", "w") as f:
        json.dump({"budget_MUSD": float(budget), "results": results}, f, indent=2)
    logger.info("Saved market_design_summary.")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vessel-type", default=VESSEL_TYPE,
                        choices=["container", "tanker"])
    main(parser.parse_args().vessel_type)
