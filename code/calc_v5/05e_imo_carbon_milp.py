"""
Module 05e (v4): IMO Carbon-Price Scenario MILP — Market vs Planning
====================================================================
Contrasts the MARKET mechanism (IMO carbon price) against CENTRAL
PLANNING (M^G): under carbon price c, each corridor's annual NET PROFIT
is (c − p*_e)·ΔE_e (p*_e = break-even price from Module 06); the MILP then
maximizes total net profit instead of abatement, so corridors that are not
economically viable at price c are simply not attractive (never forced, no
feasibility filter needed — the objective handles it).

Scenarios: c ∈ {380 (IMO proposal cap), 1000, 2000, 5000} USD/tCO2, plus
the M^G reference (abatement-maximizing, same 20% budget, methanol).
Same M_ell pruning (top-15) and budget convention as Module 05 for
comparability.

Outputs (02_数据_output/policy_analysis_v4/):
  imo_carbon_summary_{vt}.csv
  imo_carbon_activated_{c}_{vt}.csv, imo_carbon_ports_{c}_{vt}.csv
"""

import sys
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import PROCESSED_DIR, POLICY_DIR, RESULTS_DIR, MILP_PARAMS, VESSEL_TYPE
import importlib
m05 = importlib.import_module("05_milp_solving")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CARBON_PRICES = [380, 1000, 2000, 5000]
FUEL = "green_methanol"


def solve_net_profit_milp(corridors, ports, M_ell_df, budget, carbon_price,
                          bp_map, time_limit=None, mip_gap=None):
    """Maximize net profit Σ (c − p*)·ΔE·z under the standard siting
    constraints (endpoints, intermediate, budget)."""
    time_limit = time_limit or MILP_PARAMS["time_limit_s"]
    mip_gap = mip_gap or MILP_PARAMS["mip_gap"]

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

    # net profit per corridor at carbon price c
    profit = {}
    for _, row in corr.iterrows():
        cid = row["corridor_id"]
        b = bp_map.get(cid)
        if b is None or b["p_star_usd"] is None or np.isnan(b["p_star_usd"]):
            profit[cid] = -1e18  # no break-even data → never selected
        else:
            profit[cid] = (carbon_price - b["p_star_usd"]) * b["dE_e"]

    import pulp
    prob = pulp.LpProblem("IMO_Carbon_MILP", pulp.LpMaximize)
    y = {p: pulp.LpVariable(f"y_{p}", cat="Binary") for p in port_list}
    z = {cid: pulp.LpVariable(f"z_{cid}", cat="Binary")
         for cid in corr["corridor_id"]}

    prob += pulp.lpSum(profit[cid] * z[cid] for cid in z), "Net_Profit"
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
                prob += (z[cid] * k_e <= pulp.lpSum(y[p] for p in inter),
                         f"zInter_{cid}")
            else:
                prob += z[cid] == 0, f"NoInter_{cid}"

    try:
        solver = pulp.GUROBI(msg=0, timeLimit=time_limit, MIPGap=mip_gap)
        prob.solve(solver)
    except Exception:
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit, gapRel=mip_gap)
        prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    obj_val = pulp.value(prob.objective) if prob.status == 1 else 0.0
    selected = [p for p in port_list if y[p].varValue and y[p].varValue > 0.5]
    activated = [cid for cid in z if z[cid].varValue and z[cid].varValue > 0.5]

    # realized abatement (secondary metric)
    W = corr.set_index("corridor_id")["W_e"].to_dict()
    abatement = sum(W.get(cid, 0.0) for cid in activated)

    return {"status": status, "net_profit_usd": obj_val,
            "abatement_tco2e": abatement,
            "n_ports": len(selected), "n_corridors": len(activated),
            "selected_ports": selected, "activated_corridors": activated,
            "carbon_price": carbon_price}


def main(vessel_type: str):
    paths_dir = PROCESSED_DIR
    corridors = pd.read_parquet(paths_dir / f"corridors_{vessel_type}.parquet")
    ports = pd.read_parquet(paths_dir / f"candidate_ports_{vessel_type}.parquet")
    M_ell_df = pd.read_parquet(paths_dir / f"M_ell_{vessel_type}.parquet")
    bp = pd.read_csv(POLICY_DIR / f"breakeven_prices_{vessel_type}.csv")
    bp_map = bp.set_index("corridor_id").to_dict("index")
    logger.info(f"  Loaded: {len(corridors)} corridors, {len(ports)} ports, "
                f"{len(bp)} break-even prices")

    M_ell_pruned = m05.prune_mell(M_ell_df, corridors, ports,
                                  K=MILP_PARAMS["m_ell_prune_k"])
    total_cost = ports.loc[ports["is_feasible"], "cost_p"].sum()
    budget = total_cost * 0.20
    logger.info(f"  Budget (20%): {budget:.1f} M USD/yr")

    # M^G reference (abatement-maximizing)
    ref = m05.solve_milp(corridors, ports, M_ell_pruned, budget, FUEL)
    logger.info(f"  M^G reference: {ref['objective_tco2e']/1e6:.2f} Mt, "
                f"{ref['n_corridors_activated']} corridors")

    rows = [{"carbon_price": "M^G (planning)", "net_profit_usd": np.nan,
             "abatement_tco2e": ref["objective_tco2e"],
             "n_ports": ref["n_ports_selected"],
             "n_corridors": ref["n_corridors_activated"]}]
    for c in CARBON_PRICES:
        t0 = time.time()
        res = solve_net_profit_milp(corridors, ports, M_ell_pruned, budget, c, bp_map)
        rows.append({"carbon_price": c, "net_profit_usd": res["net_profit_usd"],
                     "abatement_tco2e": res["abatement_tco2e"],
                     "n_ports": res["n_ports"],
                     "n_corridors": res["n_corridors"]})
        logger.info(f"  c={c}: profit ${res['net_profit_usd']/1e9:.2f}B, "
                    f"abatement {res['abatement_tco2e']/1e6:.2f} Mt, "
                    f"{res['n_corridors']} corridors, "
                    f"{res['status']} ({time.time()-t0:.0f}s)")
        if res["activated_corridors"]:
            act = corridors[corridors["corridor_id"].isin(res["activated_corridors"])]
            act.to_csv(POLICY_DIR / f"imo_carbon_activated_{c}_{vessel_type}.csv",
                       index=False)
        if res["selected_ports"]:
            sel = ports[ports["port_code"].isin(res["selected_ports"])]
            sel.to_csv(POLICY_DIR / f"imo_carbon_ports_{c}_{vessel_type}.csv",
                       index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(POLICY_DIR / f"imo_carbon_summary_{vessel_type}.csv", index=False)
    logger.info("  Saved imo_carbon_summary.")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vessel-type", default=VESSEL_TYPE)
    args = parser.parse_args()
    main(args.vessel_type)
