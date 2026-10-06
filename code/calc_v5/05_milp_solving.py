"""
Module 05 (v2): MILP Port Siting & Corridor Activation
=======================================================
Solves the global-planner siting MILP (Eq. 13–19) with v2 conventions:

  * Physical port costs c_p (M USD/yr) from Module 04 (Eq. 17).
  * Budget B = budget_fraction × Σ_p c_p (relative budget, comparable to v1).
  * W_e recomputed per fuel scenario from corridor eta columns (no stale
    cross-scenario W_e leakage).
  * Feasibility mask (Eq. 18): y_p variables exist only for P^feas.
  * Intermediate constraint in aggregated form: z_e·k_e ≤ Σ_{p∈M_ℓ} y_p
    with M_ℓ filtered by fuel (Eq. 15, v2 sequence-based sets).

Outputs (02_数据_output/milp_results_v2/):
  pareto_{fuel}_{vt}.csv, selected_ports_{vt}.csv, activated_corridors_{vt}.csv
"""

import sys
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PROCESSED_DIR, RESULTS_DIR, EMISSION_PARAMS, MILP_PARAMS,
    VESSEL_TYPE, get_data_paths,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Shared helpers (imported by 05b/05c/05d/09)
# ============================================================
def prune_mell(M_ell_df, corridors, ports, K=15):
    """Prune intermediate candidates to top-K per (corridor, fuel),
    ranked by minimum deviation from the great-circle route (v1 logic)."""
    if len(M_ell_df) == 0:
        return M_ell_df
    port_xy = ports.drop_duplicates("port_code").set_index("port_code")[
        ["port_lon", "port_lat"]].to_dict("index")
    corr_lookup = corridors.set_index("corridor_id")[
        ["origin_lon", "origin_lat", "dest_lon", "dest_lat"]].to_dict("index")

    def deviation(row):
        c = corr_lookup.get(row["corridor_id"])
        p = port_xy.get(row["intermediate_port"])
        if c is None or p is None:
            return 1e9
        mid_lon = (c["origin_lon"] + c["dest_lon"]) / 2
        mid_lat = (c["origin_lat"] + c["dest_lat"]) / 2
        return ((p["port_lon"] - mid_lon) ** 2 + (p["port_lat"] - mid_lat) ** 2) ** 0.5

    df = M_ell_df.copy()
    df["deviation"] = df.apply(deviation, axis=1)
    pruned = (df.sort_values("deviation")
              .groupby(["corridor_id", "fuel"]).head(K).reset_index(drop=True))
    logger.info(f"  Pruned M_ell: {len(M_ell_df):,} → {len(pruned):,} entries")
    return pruned


def prepare_corridors(corridors, fuel):
    """Prepare corridor frame with fuel-specific W_e and range logic."""
    corr = corridors.copy()
    eta_col = f"eta_{fuel}" if f"eta_{fuel}" in corridors.columns else None
    if eta_col:
        corr["W_e"] = corr["E_e"] * corr[eta_col] * corr["alpha_e"]
    else:
        corr["W_e"] = corr["E_e"] * 0.8 * corr["alpha_e"]
    R_f = EMISSION_PARAMS["fuel_range_nm"].get(fuel, 5000)
    corr["range_violating"] = corr["mean_distance"] > R_f
    kcol = f"k_e_{fuel}" if f"k_e_{fuel}" in corr.columns else None
    if kcol is not None:
        corr["k_e"] = corr[kcol]
    else:
        corr["k_e"] = (np.ceil(corr["mean_distance"] / R_f).astype(int) - 1).clip(lower=0)
    return corr


def solve_milp(corridors, ports, M_ell_df, budget, fuel="green_methanol",
               solver_name="gurobi", time_limit=None, mip_gap=None):
    """Solve the GSC siting MILP (Eq. 13–19) for a given budget."""
    time_limit = time_limit or MILP_PARAMS["time_limit_s"]
    mip_gap = mip_gap or MILP_PARAMS["mip_gap"]

    corr = prepare_corridors(corridors, fuel)
    fuel_range = EMISSION_PARAMS["fuel_range_nm"].get(fuel, 5000)

    # Feasible port set only (Eq. 18)
    ports_f = ports[ports["is_feasible"]].copy()
    port_list = ports_f["port_code"].tolist()
    port_cost = ports_f.set_index("port_code")["cost_p"].to_dict()
    port_idx = {p: i for i, p in enumerate(port_list)}
    logger.info(f"  MILP({fuel}, B={budget:.1f}): {len(port_list)} feasible ports")

    # Fuel-specific intermediates
    mell_fuel = M_ell_df[M_ell_df["fuel"] == fuel] if len(M_ell_df) else M_ell_df
    corr_intermediates = {}
    if len(mell_fuel) > 0:
        for cid, g in mell_fuel.groupby("corridor_id"):
            corr_intermediates[cid] = [p for p in g["intermediate_port"].tolist()
                                       if p in port_idx]

    import pulp
    prob = pulp.LpProblem("GSC_Planning_v2", pulp.LpMaximize)
    y = {p: pulp.LpVariable(f"y_{p}", cat="Binary") for p in port_list}
    z = {cid: pulp.LpVariable(f"z_{cid}", cat="Binary")
         for cid in corr["corridor_id"]}

    # Objective (Eq. 13)
    W = corr.set_index("corridor_id")["W_e"].to_dict()
    prob += pulp.lpSum(W[cid] * z[cid] for cid in z), "Total_Abatement"

    # Budget (Eq. 16)
    prob += (pulp.lpSum(port_cost[p] * y[p] for p in port_list) <= budget, "Budget")

    # Corridor constraints (Eq. 14–15)
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

    # Solve
    if solver_name == "gurobi":
        try:
            solver = pulp.GUROBI(msg=0, timeLimit=time_limit, MIPGap=mip_gap)
            prob.solve(solver)
        except Exception:
            solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit, gapRel=mip_gap)
            prob.solve(solver)
    else:
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit, gapRel=mip_gap)
        prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    obj_val = pulp.value(prob.objective) if prob.status == 1 else 0.0
    selected = [p for p in port_list if y[p].varValue and y[p].varValue > 0.5]
    activated = [cid for cid in z if z[cid].varValue and z[cid].varValue > 0.5]

    return {"status": status, "objective_tco2e": obj_val,
            "n_ports_selected": len(selected), "n_corridors_activated": len(activated),
            "selected_ports": selected, "activated_corridors": activated,
            "budget": budget, "fuel": fuel}


def solve_pareto_frontier(corridors, ports, M_ell_df, fuel, solver_name="gurobi"):
    total_cost = ports.loc[ports["is_feasible"], "cost_p"].sum()
    results = []
    for frac in MILP_PARAMS["budget_fractions"]:
        budget = total_cost * frac
        logger.info(f"\n  Budget fraction={frac:.0%} (B={budget:.1f} M USD/yr)")
        t0 = time.time()
        res = solve_milp(corridors, ports, M_ell_df, budget, fuel, solver_name)
        res["budget_fraction"] = frac
        res["solve_time_s"] = round(time.time() - t0, 1)
        results.append(res)
        logger.info(f"    → {res['n_corridors_activated']} corridors, "
                    f"{res['objective_tco2e']/1e6:.2f} Mt, {res['solve_time_s']}s, "
                    f"{res['status']}")
    return results


def main(vessel_type: str, solver_name: str = "gurobi"):
    paths = get_data_paths(vessel_type)
    logger.info(f"MILP Solving v2: {paths['label_cn']}")

    corridors = pd.read_parquet(PROCESSED_DIR / f"corridors_{vessel_type}.parquet")
    ports = pd.read_parquet(PROCESSED_DIR / f"candidate_ports_{vessel_type}.parquet")
    M_ell_df = pd.read_parquet(PROCESSED_DIR / f"M_ell_{vessel_type}.parquet")
    logger.info(f"  Loaded: {len(corridors)} corridors, {len(ports)} ports, "
                f"{len(M_ell_df)} M_ell entries")

    M_ell_pruned = prune_mell(M_ell_df, corridors, ports,
                              K=MILP_PARAMS["m_ell_prune_k"])
    total_cost = ports.loc[ports["is_feasible"], "cost_p"].sum()
    logger.info(f"  Feasible cost pool: {total_cost:.1f} M USD/yr")

    # ---- Pareto frontier (methanol reference + ammonia comparison) ----
    for fuel in ["green_methanol", "green_ammonia"]:
        logger.info(f"\n{'─' * 50}\nFuel: {fuel}\n{'─' * 50}")
        results = solve_pareto_frontier(corridors, ports, M_ell_pruned, fuel,
                                        solver_name)
        res_df = pd.DataFrame([{
            "fuel": r["fuel"], "budget_fraction": r["budget_fraction"],
            "budget": r["budget"], "n_ports": r["n_ports_selected"],
            "n_corridors": r["n_corridors_activated"],
            "objective_tco2e": r["objective_tco2e"],
            "solve_time_s": r["solve_time_s"], "status": r["status"],
        } for r in results])
        res_df.to_csv(RESULTS_DIR / f"pareto_{fuel}_{vessel_type}.csv", index=False)
        logger.info(f"  Saved: pareto_{fuel}_{vessel_type}.csv")

    # ---- Detailed 20% budget solution (methanol, reference) ----
    logger.info(f"\n{'=' * 60}\nDETAILED SOLUTION (20% budget, methanol)\n{'=' * 60}")
    detail = solve_milp(corridors, ports, M_ell_pruned, total_cost * 0.20,
                        "green_methanol", solver_name)
    logger.info(f"  Status: {detail['status']}; ports: {detail['n_ports_selected']}; "
                f"corridors: {detail['n_corridors_activated']}; "
                f"abatement: {detail['objective_tco2e']/1e6:.2f} Mt")

    if detail["selected_ports"]:
        sel = ports[ports["port_code"].isin(detail["selected_ports"])].copy()
        sel.to_csv(RESULTS_DIR / f"selected_ports_{vessel_type}.csv", index=False)
        logger.info(f"  Saved: selected_ports_{vessel_type}.csv ({len(sel)} ports)")
    if detail["activated_corridors"]:
        act = corridors[corridors["corridor_id"].isin(detail["activated_corridors"])]
        act.to_csv(RESULTS_DIR / f"activated_corridors_{vessel_type}.csv", index=False)
        logger.info(f"  Saved: activated_corridors_{vessel_type}.csv "
                    f"({len(act)} corridors)")

    logger.info("\n✓ MILP solving v2 complete.")
    return detail


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vessel-type", default=VESSEL_TYPE)
    parser.add_argument("--solver", default="gurobi", choices=["gurobi", "cbc"])
    args = parser.parse_args()
    main(args.vessel_type, args.solver)
