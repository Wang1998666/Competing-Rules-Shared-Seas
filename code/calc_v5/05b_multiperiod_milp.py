"""
Module 05b (v2): Multi-Period Dynamic MILP — Joint Formulation
===============================================================
Solves the 6-period (2025–2050) dynamic siting model (Eq. 29–36) as ONE
joint MILP (not period-by-period decomposition):

    max  Σ_t δ^t Σ_e W_e^t z_e^t                         (Eq. 29)
    s.t. z_e^t ≤ y_{o(e)}^t,  z_e^t ≤ y_{d(e)}^t         (Eq. 30)
         z_e^t·k_e ≤ Σ_{p∈M_ℓ(e,f(t))} y_p^t             (Eq. 31)
         Σ_p c_p^t·(y_p^t − y_p^{t−1}) ≤ B^t             (Eq. 32)
         y_p^t ≥ y_p^{t−1}                                (Eq. 33)
         Σ_p φ_f·y_p^t ≤ S_f^t,  f = f(t)                (Eq. 34, v2: NEW)
         y_p^t = 0 ∀ p ∉ P^feas                          (Eq. 35)

  * Fuel availability windows: F^1={LNG}, F^2={LNG,methanol},
    F^{3+}={LNG,methanol,ammonia}; the corridor objective weight uses the
    best available fuel's η (η^t_e = max over F^t), and the fuel-supply
    constraint (Eq. 34) is imposed for the dominant new fuel of period t.
  * Learning curve c_p^t = c_p^0·(N^{t−1}+1)^{−β} with N^{t−1} cumulative
    installations — the learning factor is evaluated on the realized
    cumulative installation count (PWL of the Gurobi v2 implementation,
    reported per period).

v5: Gurobi → pulp + CBC (license expired); model semantics unchanged.

Outputs (02_数据_output/milp_results_v5/):
  multiperiod_summary_{vt}.csv, multiperiod_ports/corridors_{period}_{vt}.csv
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
    MULTIPERIOD_PARAMS, VESSEL_TYPE, get_data_paths,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_period_params():
    """Period-specific parameters (fuel windows, scenarios, budgets, caps)."""
    periods = MULTIPERIOD_PARAMS["periods"]
    r = MILP_PARAMS["discount_rate"]
    delta = {t: 1.0 / (1 + r) ** ((t - periods[0]) / 5) for t in periods}
    fuel_avail_year = MULTIPERIOD_PARAMS["fuel_availability"]
    fuel_available = {}
    for t in periods:
        fuel_available[t] = [f for f, y in fuel_avail_year.items() if t >= y]
    # dominant new fuel per period (for the supply constraint, Eq. 34)
    dominant = {}
    for t in periods:
        if t >= fuel_avail_year["green_ammonia"]:
            dominant[t] = "green_ammonia"
        elif t >= fuel_avail_year["green_methanol"]:
            dominant[t] = "green_methanol"
        else:
            dominant[t] = None  # LNG: mature supply, no cap
    return {"periods": periods, "delta": delta, "fuel_available": fuel_available,
            "dominant_fuel": dominant}


def pwl_learning_factor(n_cum, xs, ys):
    """PWL evaluation of L = (N+1)^(-β) at integer N (matches the Gurobi
    addGenConstrPWL semantics of the v2 implementation)."""
    if n_cum <= xs[0]:
        return ys[0]
    for k in range(len(xs) - 1):
        if n_cum <= xs[k + 1]:
            frac = (n_cum - xs[k]) / (xs[k + 1] - xs[k])
            return ys[k] + frac * (ys[k + 1] - ys[k])
    return ys[-1]


def solve_multiperiod_joint(corridors, ports, M_ell_df, vessel_type,
                            time_limit=1800, mip_gap=0.005, beta=None,
                            supply_scale=1.0, apply_learning=True):
    """Joint 6-period MILP with fuel supply caps (pulp + CBC).

    apply_learning=True implements Eq. 32 exactly as stated in the paper:
    the period-t budget charges learning-adjusted costs
        Σ_p c_p^0·(N^{t-1}_cum + 1)^{-β}·(y_p^t − y_p^{t-1}) ≤ B^t.
    Since L(N) = (N+1)^{-β} multiplies the whole new-build spend C_t, the
    constraint is equivalent to C_t ≤ B^t·(N_t + 1)^{β}, whose right-hand
    side is concave in N_t (0 < β < 1); it is imposed through a disjunctive
    piecewise-linear chord formulation with one binary per grid segment
    (exact at breakpoints, conservative between them because chords of a
    concave function lie below it).
    """
    import pulp
    params = get_period_params()
    periods = params["periods"]
    beta = MULTIPERIOD_PARAMS["learning_exponent_beta"] if beta is None else beta
    budget_shares = MULTIPERIOD_PARAMS["budget_shares"]

    # feasible ports
    ports_f = ports[ports["is_feasible"]].copy()
    port_list = ports_f["port_code"].tolist()
    port_cost0 = ports_f.set_index("port_code")["cost_p"].to_dict()
    n_ports = len(port_list)
    logger.info(f"  Feasible ports: {n_ports}")

    total_cost = ports_f["cost_p"].sum()
    period_budgets = {t: total_cost * 0.20 * budget_shares[i]
                      for i, t in enumerate(periods)}
    logger.info(f"  Cost pool: {total_cost:.1f} M USD/yr; "
                f"period budgets: {[round(period_budgets[t],1) for t in periods]}")

    # corridor preparation per period
    corr = corridors.copy()
    corr_ids = corr["corridor_id"].tolist()
    corr_fuel_cols = {}
    for fuel in ["LNG", "green_methanol", "green_ammonia"]:
        c = f"eta_{fuel}"
        if c in corr.columns:
            corr_fuel_cols[fuel] = corr[c].values
    emission_sc = MULTIPERIOD_PARAMS["emission_scenario"]
    adoption_sc = MULTIPERIOD_PARAMS["adoption_scenario"]

    # eta of best available fuel per period (fuel windows)
    eta_t = {}
    for t in periods:
        favail = params["fuel_available"][t]
        best = min(favail, key=lambda f: EMISSION_PARAMS["ef_wtw"][
            "LNG_HP" if f == "LNG" else f])
        eta_t[t] = corr_fuel_cols[best]

    # fuel-specific intermediate sets (pruned)
    mell = M_ell_df
    mell_dict_fuel = {}
    if len(mell) > 0:
        for (cid, fuel), g in mell.groupby(["corridor_id", "fuel"]):
            mell_dict_fuel[(cid, fuel)] = g["intermediate_port"].tolist()

    # PWL breakpoints clamped to the feasible port count (deduplicated)
    pts = MULTIPERIOD_PARAMS["learning_pwl_points"]
    xs = sorted({min(x, n_ports) for x in pts})
    ys = [(x + 1) ** (-beta) for x in xs]

    prob = pulp.LpProblem("GSC_MP_joint_v2", pulp.LpMaximize)

    y = {(t, p): pulp.LpVariable(f"y_{t}_{p}", cat="Binary")
         for t in periods for p in port_list}
    z = {(t, cid): pulp.LpVariable(f"z_{t}_{cid}", cat="Binary")
         for t in periods for cid in corr_ids}

    # Objective (Eq. 29): W_e^t = E_e · emission_scenario(t) · η^t_e · α^t
    obj_terms = []
    for i, t in enumerate(periods):
        W_t = corr["E_e"].values * emission_sc[t] * eta_t[t] * adoption_sc[t]
        w_map = dict(zip(corr_ids, W_t))
        obj_terms.append(params["delta"][t] * pulp.lpSum(
            w_map[cid] * z[t, cid] for cid in corr_ids))
    prob += pulp.lpSum(obj_terms), "obj"

    # fuel supply caps (Eq. 34) — dominant fuel per period
    for i, t in enumerate(periods):
        dfuel = params["dominant_fuel"][t]
        if dfuel is None:
            continue
        cap = MULTIPERIOD_PARAMS["fuel_supply_cap_Mt"][dfuel].get(t)
        if cap is None:
            continue
        phi = MULTIPERIOD_PARAMS["fuel_throughput_Mt_per_port"][dfuel]
        prob += (pulp.lpSum(y[t, p] for p in port_list) * phi <= cap * supply_scale,
                 f"supply_{t}_{dfuel}")

    # corridor constraints (Eq. 30–31), precomputed for speed
    port_idx = {p: i for i, p in enumerate(port_list)}
    range_info = {}
    for cid, row in corr.iterrows():
        o, d = row["origin_port"], row["dest_port"]
        info = {"o": o, "d": d, "o_ok": o in port_idx, "d_ok": d in port_idx}
        for t in periods:
            dfuel = params["dominant_fuel"][t]
            R_f = EMISSION_PARAMS["fuel_range_nm"].get(dfuel, 8000)
            if row["mean_distance"] > R_f:
                k_e = max(1, int(np.ceil(row["mean_distance"] / R_f) - 1))
                inter = [p for p in mell_dict_fuel.get((cid, dfuel), [])
                         if p in port_idx]
                info[(t, "k")] = k_e if len(inter) >= k_e else None
                info[(t, "inter")] = inter
            else:
                info[(t, "k")] = 0
        range_info[cid] = info

    for cid, info in range_info.items():
        for i, t in enumerate(periods):
            if not (info["o_ok"] and info["d_ok"]):
                prob += (z[t, cid] == 0, f"noport_{cid}_{t}")
                continue
            prob += (z[t, cid] <= y[t, info["o"]], f"zYo_{cid}_{t}")
            prob += (z[t, cid] <= y[t, info["d"]], f"zYd_{cid}_{t}")
            k_e = info[(t, "k")]
            if k_e is None:
                prob += (z[t, cid] == 0, f"nointer_{cid}_{t}")
            elif k_e > 0:
                prob += (z[t, cid] * k_e <=
                         pulp.lpSum(y[t, p] for p in info[(t, "inter")]),
                         f"zInter_{cid}_{t}")

    # budget (Eq. 32): Σ_p c_p·(y^t−y^{t−1}) ≤ B^t. With apply_learning,
    # c_p^t = c_p^0·(N^{t−1}+1)^{−β} enters the budget through the disjunctive
    # chord form C_t ≤ B^t·(N_t+1)^β described in the docstring. Without it
    # (legacy behaviour), the base costs are charged and the learning factor
    # is reported only.
    learn_grid = sorted({min(x, n_ports) for x in
                         [0, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256,
                          384, 512, 700, 10**6]})
    M_C = float(total_cost)
    M_N = float(n_ports)
    for i, t in enumerate(periods):
        if i == 0:
            expr = pulp.lpSum(port_cost0[p] * y[t, p] for p in port_list)
            prob += (expr <= period_budgets[t], f"budget_{t}")
            continue
        prev_t = periods[i - 1]
        C_t = pulp.lpSum(port_cost0[p] * (y[t, p] - y[prev_t, p])
                         for p in port_list)
        if not apply_learning:
            prob += (C_t <= period_budgets[t], f"budget_{t}")
            continue
        N_t = pulp.lpSum(y[prev_t, p] for p in port_list)
        seg_bin = [pulp.LpVariable(f"seg_{t}_{k}", cat="Binary")
                   for k in range(len(learn_grid) - 1)]
        prob += (pulp.lpSum(seg_bin) == 1, f"segsum_{t}")
        for k in range(len(learn_grid) - 1):
            x0, x1 = learn_grid[k], learn_grid[k + 1]
            if x1 == x0:
                continue
            f0, f1 = (x0 + 1) ** beta, (x1 + 1) ** beta
            chord = f0 + (f1 - f0) * (N_t - x0) / (x1 - x0)
            prob += (C_t <= period_budgets[t] * chord + M_C * (1 - seg_bin[k]),
                     f"budget_{t}_seg{k}")
            # level forcing with tight big-Ms (N_t ∈ [0, n_ports])
            prob += (N_t >= x0 * seg_bin[k], f"Nlo_{t}_seg{k}")
            prob += (N_t <= x1 + (n_ports - x1) * (1 - seg_bin[k]),
                     f"Nhi_{t}_seg{k}")

    # irreversibility (Eq. 33)
    for i in range(1, len(periods)):
        for p in port_list:
            prob += (y[periods[i], p] >= y[periods[i - 1], p],
                     f"irrev_{p}_{periods[i]}")

    # ---- Solve ----
    logger.info(f"  Solving joint MILP ({len(periods)} periods, "
                f"{len(periods) * (n_ports + len(corr))} binaries)...")
    t0 = time.time()
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit, gapRel=mip_gap)
    prob.solve(solver)
    elapsed = time.time() - t0
    status = pulp.LpStatus[prob.status]
    has_sol = any(var.varValue is not None for var in z.values())
    if not has_sol or prob.status in (pulp.LpStatusInfeasible,
                                      pulp.LpStatusUnbounded):
        logger.warning(f"  Joint MILP failed (status {status}) — "
                       f"falling back to sequential decomposition")
        return solve_multiperiod_sequential(corridors, ports, M_ell_df,
                                            vessel_type, beta=beta)

    logger.info(f"  Solved in {elapsed:.0f}s, status={status}, "
                f"obj={pulp.value(prob.objective)/1e6:.2f} Mt")
    # CBC stopping on the time limit without an incumbent writes the LP
    # relaxation into the solution file; reject fractional solutions.
    max_frac = max(abs((v.varValue or 0.0) - round(v.varValue or 0.0))
                   for v in list(y.values()) + list(z.values()))
    if max_frac > 1e-6:
        logger.warning(f"  Fractional solution (max deviation {max_frac:.3f}) — "
                       f"falling back to sequential decomposition")
        return solve_multiperiod_sequential(corridors, ports, M_ell_df,
                                            vessel_type, beta=beta)

    # ---- Extract ----
    results = []
    cum_abate = 0.0
    for i, t in enumerate(periods):
        sel = [p for p in port_list if y[t, p].varValue and y[t, p].varValue > 0.5]
        act = [cid for cid in corr_ids
               if z[t, cid].varValue and z[t, cid].varValue > 0.5]
        W_t = (corr["E_e"].values * emission_sc[t] * eta_t[t] * adoption_sc[t])
        abate = W_t[np.isin(corr["corridor_id"], act)].sum()
        cum_abate += abate
        prev = set(results[-1]["selected_ports"]) if results else set()
        newp = [p for p in sel if p not in prev]
        # learning factor at the realized cumulative installation count
        # (stock at the END of the previous period, not a sum over periods)
        n_cum_real = len(results[i - 1]["selected_ports"]) if i > 0 else 0
        learn_real = (n_cum_real + 1) ** (-beta)
        spend_base = sum(port_cost0[p] for p in newp)
        spend_learn = spend_base * learn_real
        results.append({
            "period": t, "status": status, "n_ports_total": len(sel),
            "n_ports_new": len(newp), "n_corridors_activated": len(act),
            "period_abatement_tco2e": float(abate),
            "cumulative_abatement_tco2e": float(cum_abate),
            "learning_factor": float(learn_real),
            "n_cum_prior": n_cum_real,
            "period_spend_base_musd": float(spend_base),
            "period_spend_learn_musd": float(spend_learn),
            "period_budget_musd": float(period_budgets[t]),
            "period_budget_utilization_pct": float(
                100.0 * spend_learn / period_budgets[t]),
            "selected_ports": sel, "activated_corridors": act,
        })
        logger.info(f"  {t}: {len(sel)} ports (+{len(newp)}), {len(act)} corridors, "
                    f"{abate/1e6:.2f} Mt (cum {cum_abate/1e6:.2f} Mt), "
                    f"spend {spend_learn:.0f}/{period_budgets[t]:.0f} M USD "
                    f"({100*spend_learn/period_budgets[t]:.0f}%)")

    return results


def solve_multiperiod_sequential(corridors, ports, M_ell_df, vessel_type,
                                 solver_name="cbc", time_limit_per_period=300,
                                 beta=None):
    """Fallback: sequential per-period decomposition with irreversibility
    (same conventions as the joint model; learning factor from cumulative
    installed ports of the previous period)."""
    import pulp
    params = get_period_params()
    periods = params["periods"]
    beta = MULTIPERIOD_PARAMS["learning_exponent_beta"] if beta is None else beta
    budget_shares = MULTIPERIOD_PARAMS["budget_shares"]

    ports_f = ports[ports["is_feasible"]].copy()
    port_list = ports_f["port_code"].tolist()
    port_cost0 = ports_f.set_index("port_code")["cost_p"].to_dict()
    port_idx = {p: i for i, p in enumerate(port_list)}
    total_cost = ports_f["cost_p"].sum()
    period_budgets = {t: total_cost * 0.20 * budget_shares[i]
                      for i, t in enumerate(periods)}

    corr = corridors.copy()
    emission_sc = MULTIPERIOD_PARAMS["emission_scenario"]
    adoption_sc = MULTIPERIOD_PARAMS["adoption_scenario"]

    mell_dict_fuel = {}
    if len(M_ell_df) > 0:
        for (cid, fuel), g in M_ell_df.groupby(["corridor_id", "fuel"]):
            mell_dict_fuel[(cid, fuel)] = g["intermediate_port"].tolist()

    y_prev = {p: 0 for p in port_list}
    results = []
    cum_abate = 0.0
    for i, t in enumerate(periods):
        dfuel = params["dominant_fuel"][t]
        favail = params["fuel_available"][t]
        best = min(favail, key=lambda f: EMISSION_PARAMS["ef_wtw"][
            "LNG_HP" if f == "LNG" else f])
        eta_col = f"eta_{best}"
        R_f = EMISSION_PARAMS["fuel_range_nm"].get(dfuel, 8000)
        n_cum = sum(1 for p in port_list if y_prev[p] == 1)
        learn = (n_cum + 1) ** (-beta)
        port_cost_t = {p: port_cost0[p] * learn for p in port_list}

        W_t = (corr["E_e"].values * emission_sc[t] *
               corr[eta_col].values * adoption_sc[t])
        corr = corr.assign(W_t=W_t)

        prob = pulp.LpProblem(f"GSC_MP_{t}_v2", pulp.LpMaximize)
        y = {p: pulp.LpVariable(f"y_{p}_{t}", cat="Binary") for p in port_list}
        z = {cid: pulp.LpVariable(f"z_{cid}_{t}", cat="Binary")
             for cid in corr["corridor_id"]}
        prob += pulp.lpSum(corr.set_index("corridor_id").loc[cid, "W_t"] * z[cid]
                           for cid in z) * params["delta"][t], f"Obj_{t}"
        prob += (pulp.lpSum(port_cost_t[p] * (y[p] - y_prev[p]) for p in port_list
                            if y_prev[p] == 0) <= period_budgets[t], f"Budget_{t}")
        for p in port_list:
            if y_prev[p] == 1:
                prob += y[p] == 1, f"Irrev_{p}_{t}"
        for _, row in corr.iterrows():
            cid = row["corridor_id"]
            o, d = row["origin_port"], row["dest_port"]
            if o not in port_idx or d not in port_idx:
                prob += z[cid] == 0, f"NoPort_{cid}_{t}"
                continue
            prob += z[cid] <= y[o], f"zYo_{cid}_{t}"
            prob += z[cid] <= y[d], f"zYd_{cid}_{t}"
            if row["mean_distance"] > R_f:
                k_e = max(1, int(np.ceil(row["mean_distance"] / R_f) - 1))
                inter = [p for p in mell_dict_fuel.get((cid, dfuel), [])
                         if p in port_idx]
                if len(inter) >= k_e:
                    prob += (z[cid] * k_e <= pulp.lpSum(y[p] for p in inter),
                             f"zInter_{cid}_{t}")
                else:
                    prob += z[cid] == 0, f"NoInter_{cid}_{t}"

        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit_per_period,
                                   gapRel=0.005)
        prob.solve(solver)
        sel = [p for p in port_list if y[p].varValue and y[p].varValue > 0.5]
        act = [cid for cid in z if z[cid].varValue and z[cid].varValue > 0.5]
        abate = corr.loc[corr["corridor_id"].isin(act), "W_t"].sum()
        cum_abate += abate
        newp = [p for p in sel if y_prev[p] == 0]
        results.append({
            "period": t, "status": pulp.LpStatus[prob.status],
            "n_ports_total": len(sel), "n_ports_new": len(newp),
            "n_corridors_activated": len(act),
            "period_abatement_tco2e": float(abate),
            "cumulative_abatement_tco2e": float(cum_abate),
            "learning_factor": learn, "selected_ports": sel,
            "activated_corridors": act})
        logger.info(f"  {t}: {len(sel)} ports (+{len(newp)}), {len(act)} corridors, "
                    f"{abate/1e6:.2f} Mt (cum {cum_abate/1e6:.2f} Mt)")
        y_prev = {p: 1 if p in sel else 0 for p in port_list}
    return results


def main(vessel_type: str, solver_name: str = "cbc", beta=None,
         supply_scale=1.0, apply_learning=True, tag=""):
    paths = get_data_paths(vessel_type)
    logger.info(f"Multi-Period MILP v2 (joint): {paths['label_cn']} "
                f"[beta={beta if beta is not None else MULTIPERIOD_PARAMS['learning_exponent_beta']}, "
                f"supply_scale={supply_scale}, learning_in_budget={apply_learning}, tag='{tag}']")

    corridors = pd.read_parquet(PROCESSED_DIR / f"corridors_{vessel_type}.parquet")
    ports = pd.read_parquet(PROCESSED_DIR / f"candidate_ports_{vessel_type}.parquet")
    M_ell_df = pd.read_parquet(PROCESSED_DIR / f"M_ell_{vessel_type}.parquet")

    from importlib import import_module
    m05 = import_module("05_milp_solving")
    M_ell_pruned = m05.prune_mell(M_ell_df, corridors, ports,
                                  K=MILP_PARAMS["m_ell_prune_k"])
    logger.info(f"  Loaded: {len(corridors)} corridors, {len(ports)} ports, "
                f"{len(M_ell_pruned)} pruned M_ell entries")

    t0 = time.time()
    results = solve_multiperiod_joint(corridors, ports, M_ell_pruned, vessel_type,
                                      beta=beta, supply_scale=supply_scale,
                                      apply_learning=apply_learning)
    elapsed = time.time() - t0

    suffix = "" if not tag else f"_{tag}"
    logger.info(f"\n{'=' * 60}\nMULTI-PERIOD SUMMARY (v2{suffix})\n{'=' * 60}")
    logger.info(f"  Total solve time: {elapsed:.0f}s")
    summary_rows = []
    for r in results:
        summary_rows.append({
            "period": r["period"], "n_ports_total": r["n_ports_total"],
            "n_ports_new": r["n_ports_new"], "n_corridors": r["n_corridors_activated"],
            "period_abatement_tco2e": r["period_abatement_tco2e"],
            "cumulative_abatement_tco2e": r["cumulative_abatement_tco2e"],
            "learning_factor": r["learning_factor"],
            "period_spend_learn_musd": r.get("period_spend_learn_musd"),
            "period_budget_musd": r.get("period_budget_musd"),
            "period_budget_utilization_pct": r.get("period_budget_utilization_pct"),
            "status": r["status"]})
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / f"multiperiod_summary_{vessel_type}{suffix}.csv",
                      index=False)

    periods = MULTIPERIOD_PARAMS["periods"]
    for r in results:
        sel_df = ports[ports["port_code"].isin(r["selected_ports"])][
            ["port_code", "port_lon", "port_lat", "cost_p", "is_feasible"]].copy()
        sel_df["period"] = r["period"]
        pi = periods.index(r["period"])
        prev_set = set(results[pi - 1]["selected_ports"]) if pi > 0 else set()
        sel_df["is_new"] = sel_df["port_code"].isin(set(r["selected_ports"]) - prev_set)
        sel_df.to_csv(RESULTS_DIR / f"multiperiod_ports_{r['period']}_{vessel_type}{suffix}.csv",
                      index=False)
        act_corr = corridors[corridors["corridor_id"].isin(r["activated_corridors"])][
            ["corridor_id", "origin_port", "dest_port", "E_e", "mean_distance"]].copy()
        act_corr["period"] = r["period"]
        act_corr.to_csv(RESULTS_DIR / f"multiperiod_corridors_{r['period']}_{vessel_type}{suffix}.csv",
                        index=False)
    logger.info(f"  Saved: multiperiod_summary/ports/corridors{suffix}")
    logger.info("\n✓ Multi-period MILP v2 complete.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vessel-type", default=VESSEL_TYPE)
    parser.add_argument("--solver", default="cbc", choices=["gurobi", "cbc"])
    parser.add_argument("--beta", type=float, default=None,
                        help="learning exponent override (default: config)")
    parser.add_argument("--supply-scale", type=float, default=1.0,
                        help="multiplier on all fuel supply caps")
    parser.add_argument("--no-learning", action="store_true",
                        help="legacy behaviour: learning reported, not budgeted")
    parser.add_argument("--tag", default="",
                        help="output filename suffix (e.g. beta020)")
    args = parser.parse_args()
    main(args.vessel_type, args.solver, beta=args.beta,
         supply_scale=args.supply_scale,
         apply_learning=not args.no_learning, tag=args.tag)
