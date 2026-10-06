"""
Module 19 (v4): Complete 2x2 decomposition of the EU rule's cost
=================================================================
Fills the missing cell of the 2x2 design (paper Section 5.2):

  * M^G                      : global planner, no forcing        (37.02 Mt)
  * M^G + FuelEU forcing     : forced planner, gamma = 0         (32.11 Mt)
  * M^EU (pure)              : EU-scope objective, no forcing    (8.48 Mt)
  * M^EU + FuelEU forcing    : EU-scope objective + forcing  <-- NEW

Forcing follows the budget-aware top-N rule of 13_sensitivity
(_solve_eu_recycling): top EU ports by corridor traffic, N capped by
affordability under 80% of the base budget.

Output: 02_数据_output/policy_analysis_v4/decomposition_2x2_{vt}.json
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import PROCESSED_DIR, POLICY_DIR, MILP_PARAMS, VESSEL_TYPE
from importlib import import_module
m05 = import_module("05_milp_solving")
_03c = import_module("03c_h3_policy_coverage")
EU_EEA_COUNTRIES = _03c.EU_EEA_COUNTRIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BUDGET_FRACTION = 0.20
FUEL = MILP_PARAMS["default_fuel"]


def main(vessel_type: str):
    corridors = pd.read_parquet(PROCESSED_DIR / f"corridors_{vessel_type}.parquet")
    ports = pd.read_parquet(PROCESSED_DIR / f"candidate_ports_{vessel_type}.parquet")
    ports = ports.drop_duplicates("port_code")
    mell_raw = pd.read_parquet(PROCESSED_DIR / f"M_ell_{vessel_type}.parquet")
    mell = m05.prune_mell(mell_raw, corridors, ports, K=MILP_PARAMS["m_ell_prune_k"])
    ports_f = ports[ports["is_feasible"]].copy()

    port_list = ports_f["port_code"].tolist()
    cost_p = ports_f.set_index("port_code")["cost_p"]
    port_idx = {p: i for i, p in enumerate(port_list)}
    base_budget = cost_p.sum() * BUDGET_FRACTION

    # budget-aware top-N EU forcing (identical to 13_sensitivity._solve_eu_recycling)
    eu_codes = set(ports_f[ports_f["country_code"].isin(EU_EEA_COUNTRIES)]["port_code"])
    corr_eu_count = pd.concat([
        corridors.loc[corridors["origin_port"].isin(eu_codes), "origin_port"],
        corridors.loc[corridors["dest_port"].isin(eu_codes), "dest_port"],
    ]).value_counts()
    ranked = [c for c in corr_eu_count.index if c in port_idx]
    cum_cost = np.cumsum([cost_p[p] for p in ranked])
    n_affordable = int((cum_cost <= base_budget * 0.8).sum())
    n_force = min(MILP_PARAMS["fueleu_forced_ports"], len(ranked), max(n_affordable, 1))
    forced = set(ranked[:n_force])
    logger.info(f"Forced EU ports: {len(forced)} (top-{n_force} by traffic)")

    corr = m05.prepare_corridors(corridors, FUEL)
    corr_list = corr[["corridor_id", "origin_port", "dest_port", "E_e", "W_e",
                      "k_e", "range_violating"]].copy()
    corr_list = corr_list[corr_list["origin_port"].isin(port_idx)
                          & corr_list["dest_port"].isin(port_idx)]
    port_eu = ports_f.set_index("port_code")["country_code"].isin(EU_EEA_COUNTRIES)
    corr_list["origin_eu"] = corr_list["origin_port"].map(port_eu).fillna(False)
    corr_list["dest_eu"] = corr_list["dest_port"].map(port_eu).fillna(False)
    corr_list["eu_ets_weight"] = np.where(
        corr_list["origin_eu"] & corr_list["dest_eu"], 1.0,
        np.where(corr_list["origin_eu"] | corr_list["dest_eu"], 0.5, 0.0))
    corr_list["eu_related"] = corr_list["eu_ets_weight"] > 0

    mell_dict = {}
    mell_f = mell[mell["fuel"] == FUEL]
    for cid, g in mell_f.groupby("corridor_id"):
        mell_dict[cid] = [p for p in g["intermediate_port"].tolist() if p in port_idx]

    import pulp
    prob = pulp.LpProblem("M_EU_forced", pulp.LpMaximize)
    y = {p: pulp.LpVariable(f"y_{p}", cat="Binary") for p in port_list}
    z = {cid: pulp.LpVariable(f"z_{cid}", cat="Binary")
         for cid in corr_list["corridor_id"]}

    # EU-scope objective (M^EU pure form) + forcing constraints
    prob += pulp.lpSum(row["W_e"] * z[row["corridor_id"]]
                       for _, row in corr_list.iterrows() if row["eu_related"])
    prob += (pulp.lpSum(cost_p[p] * y[p] for p in port_list) <= base_budget, "budget")
    for p in forced:
        prob += y[p] == 1, f"fueleu_{p}"
    for _, row in corr_list.iterrows():
        cid, o, d = row["corridor_id"], row["origin_port"], row["dest_port"]
        prob += z[cid] <= y[o], f"ep1_{cid}"
        prob += z[cid] <= y[d], f"ep2_{cid}"
        if row["range_violating"]:
            k_e = max(1, int(row["k_e"]))
            m_ports = mell_dict.get(cid, [])
            if len(m_ports) >= k_e:
                prob += z[cid] * k_e <= pulp.lpSum(y[m] for m in m_ports), f"int_{cid}"
            else:
                prob += z[cid] == 0, f"nointer_{cid}"

    try:
        solver = pulp.GUROBI(msg=0, timeLimit=MILP_PARAMS["time_limit_s"],
                             MIPGap=MILP_PARAMS["mip_gap"])
        prob.solve(solver)
    except Exception:
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=MILP_PARAMS["time_limit_s"],
                                   gapRel=MILP_PARAMS["mip_gap"])
        prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    if prob.status != 1:
        raise RuntimeError(f"MILP status: {status}")

    activated = [cid for cid in z if z[cid].varValue and z[cid].varValue > 0.5]
    selected = [p for p in port_list if y[p].varValue and y[p].varValue > 0.5]
    act = corr_list[corr_list["corridor_id"].isin(set(activated))]
    w_g = act["W_e"].sum() / 1e6
    w_eu = act.loc[act["eu_related"], "W_e"].sum() / 1e6
    logger.info(f"M^EU+forcing: {len(activated)} corridors, {len(selected)} ports, "
                f"W^G={w_g:.2f} Mt, W^EU={w_eu:.2f} Mt (EU-scope objective "
                f"{pulp.value(prob.objective)/1e6:.2f} Mt)")

    pd.DataFrame({"corridor_id": activated}).to_csv(
        POLICY_DIR / f"meU_forced_activated_corridors_{vessel_type}.csv", index=False)
    pd.DataFrame({"port_code": selected}).to_csv(
        POLICY_DIR / f"meU_forced_selected_ports_{vessel_type}.csv", index=False)

    # assemble the 2x2 decomposition (all cells evaluated on the global scope)
    with open(POLICY_DIR / f"utility_matrix_{vessel_type}.json") as f:
        um = json.load(f)
    with open(POLICY_DIR / f"fueleu_comparison_{vessel_type}.json") as f:
        fc = json.load(f)
    cells = {
        "G": um["matrix_Mt"]["G"]["G"],          # global planner (Module 05, 20% budget)
        "G_force": fc["rule_obj_global_Mt"],     # forced planner, gamma=0 (Module 09)
        "EU_pure": um["matrix_Mt"]["EU"]["G"],   # M^EU pure form (utility matrix)
        "EU_force": round(w_g, 2),               # new cell
    }
    scope_effect = cells["G"] - cells["EU_pure"]
    mandate_effect = cells["G"] - cells["G_force"]
    total_effect = cells["G"] - cells["EU_force"]
    interaction = total_effect - scope_effect - mandate_effect
    out = {
        "vessel_type": vessel_type, "fuel": FUEL, "budget_fraction": BUDGET_FRACTION,
        "gamma": 0.0, "n_forced_ports": len(forced),
        "n_corridors": len(activated), "n_ports": len(selected),
        "cells_global_Mt": cells,
        "W_EU_scope_Mt": {"EU_pure": um["matrix_Mt"]["EU"]["EU"], "EU_force": round(w_eu, 2)},
        "decomposition_pp": {
            "scope_only": round(scope_effect / cells["G"] * 100, 1),
            "mandate_only": round(mandate_effect / cells["G"] * 100, 1),
            "interaction": round(interaction / cells["G"] * 100, 1),
            "total": round(total_effect / cells["G"] * 100, 1),
        },
    }
    with open(POLICY_DIR / f"decomposition_2x2_{vessel_type}.json", "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Decomposition (pp of global abatement): "
                f"scope {out['decomposition_pp']['scope_only']}, "
                f"mandate {out['decomposition_pp']['mandate_only']}, "
                f"interaction {out['decomposition_pp']['interaction']}, "
                f"total {out['decomposition_pp']['total']}")
    logger.info(f"Saved: decomposition_2x2_{vessel_type}.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vessel-type", default=VESSEL_TYPE)
    args = parser.parse_args()
    main(args.vessel_type)
