"""
Module 14 (v4): Rule-Competition Utility Matrix (3x3)
=====================================================
Re-solves the two rule-sphere MILPs on the v4 data basis and assembles the
full competition utility matrix U = [W^v(y^u)] (paper Section 4.6):

  * M^G   : global planner (Module 05 detailed solution, 20% budget)
  * M^EU  : EU rule sphere — objective restricted to EU-related corridors
            (eu_ets_weight > 0), global investment domain, no FuelEU forcing,
            gamma = 0 (exclusive-rule benchmark, same as calc_v2/09 view="eu")
  * M^BRI : BRI investment sphere — objective restricted to BRI corridors,
            investment domain restricted to BRI ports (calc_v2/05c logic)

The matrix evaluates each solution u under every rule's corridor scope v:

  W^G(y^u)   = sum of W_e over all activated corridors
  W^EU(y^u)  = sum of W_e over activated corridors in the EU ETS scope
  W^BRI(y^u) = sum of W_e over activated corridors in the BRI scope

Outputs (02_数据_output/policy_analysis_v4/):
  utility_matrix_{vt}.csv, utility_matrix_{vt}.json,
  meU_activated_corridors_{vt}.csv, meB_activated_corridors_{vt}.csv
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PROCESSED_DIR, RESULTS_DIR, POLICY_DIR, EMISSION_PARAMS, MILP_PARAMS,
    VESSEL_TYPE,
)
from importlib import import_module
m05 = import_module("05_milp_solving")
_08 = import_module("08_bri_analysis")
BRI_MARITIME_COUNTRIES = _08.BRI_MARITIME_COUNTRIES
_03c = import_module("03c_h3_policy_coverage")
EU_EEA_COUNTRIES = _03c.EU_EEA_COUNTRIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BUDGET_FRACTION = 0.20
FUEL = MILP_PARAMS["default_fuel"]


def load_base(vessel_type):
    corridors = pd.read_parquet(PROCESSED_DIR / f"corridors_{vessel_type}.parquet")
    ports = pd.read_parquet(PROCESSED_DIR / f"candidate_ports_{vessel_type}.parquet")
    ports = ports.drop_duplicates("port_code")
    mell_raw = pd.read_parquet(PROCESSED_DIR / f"M_ell_{vessel_type}.parquet")
    mell = m05.prune_mell(mell_raw, corridors, ports, K=MILP_PARAMS["m_ell_prune_k"])
    return corridors, ports, mell


def build_corr_list(corridors, ports_f, port_idx, vessel_type):
    """Fuel-specific corridor frame with governance scopes (EU weight, BRI)."""
    corr = m05.prepare_corridors(corridors, FUEL)
    corr_list = corr[["corridor_id", "origin_port", "dest_port", "E_e", "W_e",
                      "mean_distance", "k_e", "range_violating"]].copy()
    corr_list = corr_list[corr_list["origin_port"].isin(port_idx) &
                          corr_list["dest_port"].isin(port_idx)]

    port_eu = ports_f.set_index("port_code")["country_code"].isin(EU_EEA_COUNTRIES)
    corr_list["origin_eu"] = corr_list["origin_port"].map(port_eu).fillna(False)
    corr_list["dest_eu"] = corr_list["dest_port"].map(port_eu).fillna(False)
    corr_list["eu_ets_weight"] = np.where(
        corr_list["origin_eu"] & corr_list["dest_eu"], 1.0,
        np.where(corr_list["origin_eu"] | corr_list["dest_eu"], 0.5, 0.0))
    corr_list["eu_related"] = corr_list["eu_ets_weight"] > 0

    bri_corr = set(pd.read_parquet(POLICY_DIR / f"bri_corridors_{vessel_type}.parquet")
                   .loc[lambda d: d["is_bri"], "corridor_id"])
    corr_list["is_bri"] = corr_list["corridor_id"].isin(bri_corr)
    return corr_list


def add_common_constraints(prob, y, z, corr_list, mell_dict, port_idx):
    """Endpoint (Eq. 14) and aggregated intermediate (Eq. 15) constraints (pulp)."""
    import pulp
    for _, row in corr_list.iterrows():
        cid, o, d = row["corridor_id"], row["origin_port"], row["dest_port"]
        prob += z[cid] <= y[o], f"ep1_{cid}"
        prob += z[cid] <= y[d], f"ep2_{cid}"
        if row["range_violating"]:
            k_e = max(1, int(row["k_e"]))
            m_ports = mell_dict.get(cid, [])
            if len(m_ports) >= k_e:
                prob += (z[cid] * k_e <= pulp.lpSum(y[m] for m in m_ports),
                         f"int_{cid}")
            else:
                prob += z[cid] == 0, f"nointer_{cid}"


def _solve_pulp(prob, y, z):
    """Solve with CBC (Gurobi license expired); returns selected/activated."""
    import pulp
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=MILP_PARAMS["time_limit_s"],
                               gapRel=MILP_PARAMS["mip_gap"])
    prob.solve(solver)
    if pulp.LpStatus[prob.status] not in ("Optimal", "Not Solved"):
        raise RuntimeError(f"MILP status {pulp.LpStatus[prob.status]}")
    activated = [cid for cid in z if z[cid].varValue and z[cid].varValue > 0.5]
    selected = [p for p in y if y[p].varValue and y[p].varValue > 0.5]
    return activated, selected, pulp.value(prob.objective)


def solve_meu(corr_list, ports_f, mell):
    """M^EU: objective restricted to EU-related corridors (view='eu', no force)."""
    import pulp

    port_list = ports_f["port_code"].tolist()
    cost_p = ports_f.set_index("port_code")["cost_p"]
    port_idx = {p: i for i, p in enumerate(port_list)}
    base_budget = ports_f["cost_p"].sum() * BUDGET_FRACTION

    mell_dict = {}
    mell_f = mell[mell["fuel"] == FUEL]
    for cid, g in mell_f.groupby("corridor_id"):
        mell_dict[cid] = [p for p in g["intermediate_port"].tolist() if p in port_idx]

    prob = pulp.LpProblem("M_EU_v4", pulp.LpMaximize)
    y = {p: pulp.LpVariable(f"y_{p}", cat="Binary") for p in port_list}
    z = {cid: pulp.LpVariable(f"z_{cid}", cat="Binary")
         for cid in corr_list["corridor_id"]}

    prob += pulp.lpSum(row["W_e"] * z[row["corridor_id"]]
                       for _, row in corr_list.iterrows() if row["eu_related"])
    prob += (pulp.lpSum(cost_p[p] * y[p] for p in port_list) <= base_budget, "budget")
    add_common_constraints(prob, y, z, corr_list, mell_dict, port_idx)
    activated, selected, obj = _solve_pulp(prob, y, z)
    logger.info(f"  M^EU: obj={obj/1e6:.2f} Mt (EU scope), "
                f"{len(activated)} corridors, {len(selected)} ports")
    return activated, selected


def solve_mbri(corr_list, ports_f, mell):
    """M^BRI: objective restricted to BRI corridors, BRI ports only."""
    import pulp

    ports_f = ports_f.copy()
    ports_f["is_bri_port"] = ports_f["country_code"].isin(BRI_MARITIME_COUNTRIES)
    bri_ports = ports_f[ports_f["is_bri_port"]].copy()
    port_list = bri_ports["port_code"].tolist()
    cost_p = bri_ports.set_index("port_code")["cost_p"]
    port_idx = {p: i for i, p in enumerate(port_list)}
    base_budget = ports_f["cost_p"].sum() * BUDGET_FRACTION  # same total pool as M^G
    logger.info(f"  BRI feasible ports: {len(bri_ports)} / {len(ports_f)}")

    corr_bri = corr_list[corr_list["origin_port"].isin(port_idx) &
                         corr_list["dest_port"].isin(port_idx)].copy()

    mell_dict = {}
    mell_f = mell[mell["fuel"] == FUEL]
    for cid, g in mell_f.groupby("corridor_id"):
        mell_dict[cid] = [p for p in g["intermediate_port"].tolist() if p in port_idx]

    prob = pulp.LpProblem("M_BRI_v4", pulp.LpMaximize)
    y = {p: pulp.LpVariable(f"y_{p}", cat="Binary") for p in port_list}
    z = {cid: pulp.LpVariable(f"z_{cid}", cat="Binary")
         for cid in corr_bri["corridor_id"]}

    prob += pulp.lpSum(row["W_e"] * z[row["corridor_id"]]
                       for _, row in corr_bri.iterrows() if row["is_bri"])
    prob += (pulp.lpSum(cost_p[p] * y[p] for p in port_list) <= base_budget, "budget")
    add_common_constraints(prob, y, z, corr_bri, mell_dict, port_idx)
    activated, selected, obj = _solve_pulp(prob, y, z)
    logger.info(f"  M^BRI: obj={obj/1e6:.2f} Mt (BRI scope), "
                f"{len(activated)} corridors, {len(selected)} ports")
    return activated, selected


def scope_sum(corr_list, activated_ids):
    """W totals of a solution's activated set under each rule scope."""
    act = corr_list[corr_list["corridor_id"].isin(set(activated_ids))]
    return {
        "G": act["W_e"].sum() / 1e6,
        "EU": act.loc[act["eu_related"], "W_e"].sum() / 1e6,
        "BRI": act.loc[act["is_bri"], "W_e"].sum() / 1e6,
    }


def main(vessel_type: str):
    logger.info("=" * 60)
    logger.info(f"RULE-COMPETITION UTILITY MATRIX v4 ({vessel_type})")
    logger.info("=" * 60)

    corridors, ports, mell = load_base(vessel_type)
    ports_f = ports[ports["is_feasible"]].copy()
    corr_list = build_corr_list(corridors, ports_f,
                                {p: i for i, p in enumerate(ports_f["port_code"])},
                                vessel_type)
    logger.info(f"  Corridors: {len(corr_list)} "
                f"(EU scope: {corr_list['eu_related'].sum()}, "
                f"BRI scope: {corr_list['is_bri'].sum()})")

    # M^G: Module 05 detailed solution (20% budget, methanol)
    act_g = pd.read_csv(RESULTS_DIR / f"activated_corridors_{vessel_type}.csv")
    ids_g = act_g["corridor_id"].tolist()
    logger.info(f"  M^G loaded: {len(ids_g)} corridors")

    # M^EU / M^BRI: re-solve on the v4 basis
    ids_eu, sel_eu = solve_meu(corr_list, ports_f, mell)
    ids_bri, sel_bri = solve_mbri(corr_list, ports_f, mell)

    solutions = {"G": ids_g, "EU": ids_eu, "BRI": ids_bri}
    matrix = {u: scope_sum(corr_list, ids) for u, ids in solutions.items()}

    # MPC relative to M^G global benchmark
    wg_g = matrix["G"]["G"]
    mpc = {u: (wg_g - matrix[u]["G"]) / wg_g * 100 for u in solutions}

    # Persist solutions + matrix
    pd.DataFrame({"corridor_id": ids_eu}).to_csv(
        POLICY_DIR / f"meU_activated_corridors_{vessel_type}.csv", index=False)
    pd.DataFrame({"port_code": sel_eu}).to_csv(
        POLICY_DIR / f"meU_selected_ports_{vessel_type}.csv", index=False)
    pd.DataFrame({"corridor_id": ids_bri}).to_csv(
        POLICY_DIR / f"meB_activated_corridors_{vessel_type}.csv", index=False)
    pd.DataFrame({"port_code": sel_bri}).to_csv(
        POLICY_DIR / f"meB_selected_ports_{vessel_type}.csv", index=False)

    rows = []
    for u in ["G", "EU", "BRI"]:
        rows.append({
            "investing_rule": f"M^{u}",
            "n_corridors_activated": len(solutions[u]),
            "W_G_Mt": round(matrix[u]["G"], 2),
            "W_EU_Mt": round(matrix[u]["EU"], 2),
            "W_BRI_Mt": round(matrix[u]["BRI"], 2),
            "MPC_pct": round(mpc[u], 1),
        })
    mat_df = pd.DataFrame(rows)
    mat_df.to_csv(POLICY_DIR / f"utility_matrix_{vessel_type}.csv", index=False)
    with open(POLICY_DIR / f"utility_matrix_{vessel_type}.json", "w") as f:
        json.dump({
            "budget_fraction": BUDGET_FRACTION, "fuel": FUEL,
            "matrix_Mt": {u: {v: round(matrix[u][v], 2) for v in ["G", "EU", "BRI"]}
                          for u in solutions},
            "MPC_pct": {u: round(mpc[u], 1) for u in solutions},
            "n_corridors": {u: len(solutions[u]) for u in solutions},
            "bri_share_pct": {u: round(matrix[u]["BRI"] / matrix[u]["G"] * 100, 1)
                              if matrix[u]["G"] > 0 else None
                              for u in solutions},
        }, f, indent=2)

    logger.info("\nCompetition utility matrix (Mt CO2e yr-1):")
    logger.info("\n" + mat_df.to_string(index=False))
    logger.info("Saved: utility_matrix_* (v4)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vessel-type", default=VESSEL_TYPE)
    args = parser.parse_args()
    main(args.vessel_type)
