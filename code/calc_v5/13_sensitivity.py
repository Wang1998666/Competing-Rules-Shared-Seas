"""
Module 13 (v4): Sensitivity Analyses — tau / ETS price / uniform cost / alpha
=============================================================================
Fulfils the sensitivity commitments of the manuscript (§4: "Sensitivity
analyses vary the FuelEU threshold tau, the adoption share alpha_e, and the
ETS price" and the uniform-cost variant of Eq. 17).

Design (minimum recomputation):
  * tau: re-classifies from the ALREADY-STORED fueleu_max column of
    governance_corridors (no path/pressure reload, no MILP re-solve).
  * ETS price: re-solves the EU rule MILP with revenue recycling
    (Eq. 25) in pulp (Gurobi with CBC fallback), gamma x P_ETS grid.
  * uniform cost: c_p = c_bar = 16.3 M USD/yr (manuscript §4 mean) with
    the same relative-budget convention as Module 05.
  * alpha_e: uniform alpha in {0.3, 0.5, 0.7} replacing the heterogeneous
    column (non-uniform variant, requires re-solve; the homogeneous
    scaling invariance is reported in the summary).

Outputs (02_数据_output/policy_analysis_v4/, NEW files only — nothing
existing is overwritten):
  sensitivity_tau_{vt}.csv, sensitivity_ets_price_{vt}.csv,
  sensitivity_uniform_cost_{vt}.csv, sensitivity_alpha_{vt}.csv,
  sensitivity_summary_{vt}.json
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (PROCESSED_DIR, RESULTS_DIR, POLICY_DIR, H3_DIR,
                    MILP_PARAMS, VESSEL_TYPE, EMISSION_PARAMS)
from importlib import import_module
m05 = import_module("05_milp_solving")
_03c = import_module("03c_h3_policy_coverage")
EU_EEA_COUNTRIES = _03c.EU_EEA_COUNTRIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TAU_GRID = [0.10, 0.30, 0.50]
ETS_GRID = {"gamma": [0.5, 1.0], "ets_price": [60.0, 80.0, 100.0, 120.0]}
UNIFORM_COST_MUSD = 16.3   # manuscript §4 mean annualized cost (M USD/yr)
ALPHA_GRID = [0.3, 0.5, 0.7]
FUEL = "green_methanol"
BUDGET_FRAC = 0.20


def _load_inputs(vt):
    corridors = pd.read_parquet(PROCESSED_DIR / f"corridors_{vt}.parquet")
    ports = pd.read_parquet(PROCESSED_DIR / f"candidate_ports_{vt}.parquet")
    M_ell_df = pd.read_parquet(PROCESSED_DIR / f"M_ell_{vt}.parquet")
    ports = ports.drop_duplicates("port_code")
    mell = m05.prune_mell(M_ell_df, corridors, ports, K=MILP_PARAMS["m_ell_prune_k"])
    return corridors, ports, mell


# ============================================================
# 1) FuelEU threshold tau sensitivity (classification only)
# ============================================================
def run_tau_sensitivity(vt):
    gov = pd.read_parquet(POLICY_DIR / f"governance_corridors_{vt}.parquet")
    p75 = gov.loc[gov["E_e"] > 0, "E_e"].quantile(0.75)
    total_em = gov["E_e"].sum()
    rows = []
    for tau in TAU_GRID:
        exposed = gov["fueleu_max"] >= tau
        re = (gov["eu_ets_endpoint"].astype(int) + gov["is_bri"].astype(int)
              + exposed.astype(int))
        conds = [gov["eu_ets_endpoint"],
                 ~gov["eu_ets_endpoint"] & gov["is_bri"],
                 ~gov["eu_ets_endpoint"] & ~gov["is_bri"] & exposed,
                 re == 0]
        gtype = np.select(conds, ["EU_ETS", "BRI_only", "FuelEU_exposed", "Vacuum"],
                          default="Vacuum")
        is_vac = (gtype == "Vacuum") & (gov["E_e"] >= p75)
        rvi_strict = gov.loc[is_vac, "E_e"].sum() / total_em * 100.0
        rvi_any = gov.loc[gtype == "Vacuum", "E_e"].sum() / total_em * 100.0
        counts = pd.Series(gtype).value_counts().to_dict()
        rows.append({
            "tau": tau,
            "n_fueleu_exposed": counts.get("FuelEU_exposed", 0),
            "n_vacuum": counts.get("Vacuum", 0),
            "n_vacuum_highE": int(is_vac.sum()),
            "RVI_strict_pct": round(rvi_strict, 1),
            "RVI_any_pct": round(rvi_any, 1),
        })
        logger.info(f"  tau={tau}: FuelEU={counts.get('FuelEU_exposed', 0):,}, "
                    f"Vacuum={counts.get('Vacuum', 0):,}, "
                    f"highE-vacuum={int(is_vac.sum()):,}, RVI_strict={rvi_strict:.1f}%")
    out = pd.DataFrame(rows)
    out.to_csv(POLICY_DIR / f"sensitivity_tau_{vt}.csv", index=False)
    logger.info(f"  Saved: sensitivity_tau_{vt}.csv")
    return out


# ============================================================
# 2) ETS price sensitivity — EU rule MILP with recycling (pulp)
# ============================================================
def _solve_eu_recycling(corridors, ports, mell, budget_frac, gamma, ets_price,
                        force_fueleu=True):
    """EU rule-sphere MILP with ETS revenue recycling (Eq. 25), global
    objective (view=global, consistent with fueleu_comparison outputs)."""
    import pulp
    ports_f = ports[ports["is_feasible"]].copy()
    port_list = ports_f["port_code"].tolist()
    cost_p = ports_f.set_index("port_code")["cost_p"]
    port_idx = {p: i for i, p in enumerate(port_list)}
    total_cost = cost_p.sum()
    base_budget = total_cost * budget_frac

    # EU ports forced open (top-N by corridor traffic; budget-aware, v2 fix)
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

    mell_dict = {}
    mell_f = mell[mell["fuel"] == FUEL] if len(mell) else mell
    for cid, g in mell_f.groupby("corridor_id"):
        mell_dict[cid] = [p for p in g["intermediate_port"].tolist() if p in port_idx]

    prob = pulp.LpProblem("EU_rule_recycling", pulp.LpMaximize)
    y = {p: pulp.LpVariable(f"y_{p}", cat="Binary") for p in port_list}
    z = {cid: pulp.LpVariable(f"z_{cid}", cat="Binary")
         for cid in corr_list["corridor_id"]}

    W = corr_list.set_index("corridor_id")["W_e"].to_dict()
    prob += pulp.lpSum(W[cid] * z[cid] for cid in z), "Total_Abatement"

    # Budget with ETS revenue recycling (Eq. 25)
    recycle = pulp.lpSum(gamma * ets_price * row["eu_ets_weight"] * row["E_e"] / 1e6
                         * z[row["corridor_id"]]
                         for _, row in corr_list.iterrows())
    prob += (pulp.lpSum(cost_p[p] * y[p] for p in port_list) - recycle
             <= base_budget, "budget_ets_recycling")

    if force_fueleu:
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
                prob += z[cid] * k_e <= pulp.lpSum(y[p] for p in m_ports), f"int_{cid}"
            else:
                prob += z[cid] == 0, f"nointer_{cid}"

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=MILP_PARAMS["time_limit_s"],
                               gapRel=MILP_PARAMS["mip_gap"])
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    if prob.status != 1:
        return None
    selected = [p for p in port_list if y[p].varValue and y[p].varValue > 0.5]
    activated = [cid for cid in z if z[cid].varValue and z[cid].varValue > 0.5]
    obj_global = sum(W[cid] for cid in activated)
    return {"status": status, "obj_global_Mt": obj_global / 1e6,
            "n_ports": len(selected), "n_corridors": len(activated),
            "n_forced": len(forced), "base_budget_MUSD": round(base_budget, 1),
            "recycle_MUSD": round(gamma * ets_price * sum(
                row["eu_ets_weight"] * row["E_e"] for _, row in
                corr_list[corr_list["corridor_id"].isin(activated)].iterrows()) / 1e6, 1)}


def run_ets_price_sensitivity(vt):
    corridors, ports, mell = _load_inputs(vt)
    # baseline: M^G 20% budget (Module 05)
    pareto = pd.read_csv(RESULTS_DIR / f"pareto_{FUEL}_{vt}.csv")
    base_row = pareto[pareto["budget_fraction"] == BUDGET_FRAC].iloc[0]
    base_mt = base_row["objective_tco2e"] / 1e6

    rows = []
    for gamma in ETS_GRID["gamma"]:
        for ets_price in ETS_GRID["ets_price"]:
            res = _solve_eu_recycling(corridors, ports, mell, BUDGET_FRAC,
                                      gamma, ets_price)
            if res is None:
                logger.warning(f"  gamma={gamma}, P_ETS={ets_price}: infeasible/failed")
                continue
            mpc = (base_mt - res["obj_global_Mt"]) / base_mt * 100.0
            rows.append({"gamma": gamma, "ets_price_usd": ets_price,
                         "obj_global_Mt": round(res["obj_global_Mt"], 2),
                         "n_ports": res["n_ports"], "n_corridors": res["n_corridors"],
                         "n_forced": res["n_forced"],
                         "recycle_MUSD": res["recycle_MUSD"],
                         "MPC_global_pct": round(mpc, 2)})
            logger.info(f"  gamma={gamma}, P_ETS={ets_price}: "
                        f"{res['obj_global_Mt']:.2f} Mt, {res['n_ports']} ports, "
                        f"MPC={mpc:.1f}% (baseline {base_mt:.2f} Mt)")
    out = pd.DataFrame(rows)
    out.to_csv(POLICY_DIR / f"sensitivity_ets_price_{vt}.csv", index=False)
    logger.info(f"  Saved: sensitivity_ets_price_{vt}.csv (baseline {base_mt:.2f} Mt)")
    return out, base_mt


# ============================================================
# 3) Uniform-cost sensitivity (c_p = c_bar, Eq. 17 variant)
# ============================================================
def run_uniform_cost_sensitivity(vt):
    corridors, ports, mell = _load_inputs(vt)
    ports_u = ports.copy()
    n_feas = int(ports_u["is_feasible"].sum())
    ports_u["cost_p"] = UNIFORM_COST_MUSD
    logger.info(f"  Uniform cost c_bar={UNIFORM_COST_MUSD} M USD/yr "
                f"({n_feas} feasible ports)")

    het = pd.read_csv(RESULTS_DIR / f"pareto_{FUEL}_{vt}.csv")
    results = m05.solve_pareto_frontier(corridors, ports_u, mell, FUEL)
    rows = []
    for r in results:
        h = het[het["budget_fraction"] == r["budget_fraction"]].iloc[0]
        rows.append({
            "budget_fraction": r["budget_fraction"],
            "budget_MUSD": round(r["budget"], 1),
            "n_ports": r["n_ports_selected"],
            "n_corridors": r["n_corridors_activated"],
            "abatement_Mt": round(r["objective_tco2e"] / 1e6, 2),
            "abatement_Mt_hetero": round(h["objective_tco2e"] / 1e6, 2),
            "n_ports_hetero": int(h["n_ports"]),
            "status": r["status"],
        })
        logger.info(f"  frac={r['budget_fraction']:.0%}: {r['n_ports_selected']} ports, "
                    f"{r['objective_tco2e']/1e6:.2f} Mt "
                    f"(hetero: {h['objective_tco2e']/1e6:.2f} Mt)")
    out = pd.DataFrame(rows)
    out.to_csv(POLICY_DIR / f"sensitivity_uniform_cost_{vt}.csv", index=False)
    logger.info(f"  Saved: sensitivity_uniform_cost_{vt}.csv")
    return out


# ============================================================
# 4) Adoption share alpha_e sensitivity (uniform replacement)
# ============================================================
def run_alpha_sensitivity(vt):
    corridors, ports, mell = _load_inputs(vt)
    base = pd.read_csv(RESULTS_DIR / f"pareto_{FUEL}_{vt}.csv")
    rows = []
    for a in ALPHA_GRID:
        corr_a = corridors.copy()
        corr_a["alpha_e"] = a
        results = m05.solve_pareto_frontier(corr_a, ports, mell, FUEL)
        for r in results:
            rows.append({
                "alpha_e": a,
                "budget_fraction": r["budget_fraction"],
                "n_ports": r["n_ports_selected"],
                "n_corridors": r["n_corridors_activated"],
                "abatement_Mt": round(r["objective_tco2e"] / 1e6, 2),
                "status": r["status"],
            })
            if r["budget_fraction"] == BUDGET_FRAC:
                logger.info(f"  alpha={a}: {r['n_ports_selected']} ports, "
                            f"{r['objective_tco2e']/1e6:.2f} Mt")
    out = pd.DataFrame(rows)
    out.to_csv(POLICY_DIR / f"sensitivity_alpha_{vt}.csv", index=False)
    logger.info(f"  Saved: sensitivity_alpha_{vt}.csv")
    return out


# ============================================================
# 5) LNG fuel scenario (R_f = inf — range constraints vacuous)
# ============================================================
def run_lng_sensitivity(vt):
    corridors, ports, mell = _load_inputs(vt)
    logger.info("  LNG scenario (R_f = inf: range constraints drop out)")
    results = m05.solve_pareto_frontier(corridors, ports, mell, "LNG")
    rows = []
    for r in results:
        rows.append({
            "fuel": "LNG",
            "budget_fraction": r["budget_fraction"],
            "n_ports": r["n_ports_selected"],
            "n_corridors": r["n_corridors_activated"],
            "abatement_Mt": round(r["objective_tco2e"] / 1e6, 2),
            "status": r["status"],
        })
        if r["budget_fraction"] == BUDGET_FRAC:
            logger.info(f"  LNG 20%: {r['n_ports_selected']} ports, "
                        f"{r['objective_tco2e']/1e6:.2f} Mt")
    out = pd.DataFrame(rows)
    out.to_csv(POLICY_DIR / f"sensitivity_lng_{vt}.csv", index=False)
    logger.info(f"  Saved: sensitivity_lng_{vt}.csv")
    return out


def build_summary(vt, tau_df, ets_df, base_mt, uni_df, alpha_df, lng_df):
    summary = {
        "version": "v4-sensitivity",
        "vessel_type": vt,
        "fuel": FUEL,
        "budget_fraction": BUDGET_FRAC,
        "baseline_MG_Mt": round(base_mt, 2),
        "note_tau": ("tau changes the governance classification statistics only; "
                     "the MILP family does not depend on tau (EU rule uses the "
                     "ETS endpoint rule, BRI rule uses country membership)."),
        "note_alpha": ("uniform alpha replacement is a non-uniform rescaling of "
                       "W_e per corridor (alpha_e is corridor-specific in the "
                       "baseline); a HOMOGENEOUS rescaling of all alpha_e by a "
                       "constant leaves the optimal y,z unchanged and scales "
                       "abatement linearly, by linear homogeneity of the "
                       "objective in W_e."),
        "tau_rows": tau_df.to_dict("records"),
        "ets_rows": ets_df.to_dict("records"),
        "uniform_cost_rows": uni_df.to_dict("records"),
        "alpha_rows": alpha_df[alpha_df["budget_fraction"] == BUDGET_FRAC].to_dict("records"),
        "lng_rows": lng_df.to_dict("records"),
    }
    with open(POLICY_DIR / f"sensitivity_summary_{vt}.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Saved: sensitivity_summary_{vt}.json")


def main(vt):
    logger.info("=" * 60)
    logger.info(f"SENSITIVITY ANALYSES — v4 ({vt})")
    logger.info("=" * 60)

    tau_df = run_tau_sensitivity(vt)
    ets_df, base_mt = run_ets_price_sensitivity(vt)
    uni_df = run_uniform_cost_sensitivity(vt)
    alpha_df = run_alpha_sensitivity(vt)
    lng_df = run_lng_sensitivity(vt)
    build_summary(vt, tau_df, ets_df, base_mt, uni_df, alpha_df, lng_df)
    logger.info("\nDone.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vessel-type", default=VESSEL_TYPE)
    args = parser.parse_args()
    main(args.vessel_type)
