"""
Module 16 (v4): Robustness Checks — reviewer-response experiments
=================================================================
Two robustness experiments supporting manuscript claims added in the
GLM-review revision:

  A. FuelEU definition robustness: pressure-field (baseline, tau=0.3) vs
     endpoint-based alternative (either endpoint in EU/EEA — the
     jurisdictionally exact scope, since FuelEU applies to the whole voyage
     of ships calling at EU/EEA ports). Re-classifies the four governance
     types under both definitions and quantifies the corridors on which
     the two definitions disagree.

  B. Range-constraint formulation equivalence:
     * methanol: all range-violating corridors have k_e = 1, where the
       aggregated form z_e*k_e <= sum(y_p) coincides exactly with the
       per-leg form. Verified analytically from the k_e distribution and
       numerically by re-solving the global planner with the per-leg
       formulation at 10% and 20% budgets.
     * ammonia: corridors with k_e >= 2 exist; compares the aggregated
       counting constraint against the stronger chain-partition
       formulation (the selected intermediates must form a feasible
       bunkering chain with every sub-leg <= R_f) at 10% and 20% budgets.

Also verifies the absolute budget levels cited in the manuscript
(full-upgrade cost pool and the 20%/10% budgets in M USD/yr).

Outputs (02_数据_output/policy_analysis_v4/, NEW files only — nothing
existing is overwritten):
  robustness_fueleu_definition_{vt}.csv / .json
  robustness_range_constraint_{vt}.csv / .json
  robustness_summary_{vt}.json

Run:  python 16_robustness_checks.py [--vessel-type container]
"""

import sys
import json
import logging
import argparse
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (PROCESSED_DIR, RESULTS_DIR, POLICY_DIR, VESSEL_TYPE,
                    MILP_PARAMS, EMISSION_PARAMS)
from importlib import import_module
m05 = import_module("05_milp_solving")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FUEL_REF = "green_methanol"
FUEL_SHORT = "green_ammonia"
TAU = 0.30
BUDGET_FRACS = [0.10, 0.20]
GOV_TYPES = ["EU_ETS", "BRI_only", "FuelEU_exposed", "Vacuum"]
# Tighter-than-production gap so formulation differences (expected < 0.1%)
# are not confounded by solver tolerance.
ROBUSTNESS_MIP_GAP = 1e-5


def haversine_nm(lon1, lat1, lon2, lat2):
    R = 3440.065  # nautical miles
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ============================================================
# A) FuelEU definition robustness (classification level)
# ============================================================
def _classify(fueleu_flag: pd.Series, ets_endpoint: pd.Series, is_bri: pd.Series):
    re = ets_endpoint.astype(int) + is_bri.astype(int) + fueleu_flag.astype(int)
    conds = [ets_endpoint,
             ~ets_endpoint & is_bri,
             ~ets_endpoint & ~is_bri & fueleu_flag,
             re == 0]
    return np.select(conds, GOV_TYPES, default="Vacuum"), re


def run_fueleu_definition_robustness(vt):
    gov = pd.read_parquet(POLICY_DIR / f"governance_corridors_{vt}.parquet")
    total_em = gov["E_e"].sum()
    p75 = gov.loc[gov["E_e"] > 0, "E_e"].quantile(0.75)

    pressure = gov["fueleu_max"] >= TAU
    endpoint = gov["eu_ets_endpoint"]          # origin_eu | dest_eu (same column)

    rows = []
    for name, flag in [("pressure_field_tau0.3", pressure),
                       ("endpoint_based", endpoint)]:
        gtype, _ = _classify(flag, gov["eu_ets_endpoint"], gov["is_bri"])
        is_vac_high = (gtype == "Vacuum") & (gov["E_e"] >= p75)
        rvi_strict = gov.loc[is_vac_high, "E_e"].sum() / total_em * 100
        rvi_any = gov.loc[gtype == "Vacuum", "E_e"].sum() / total_em * 100
        for gt in GOV_TYPES:
            sel = gtype == gt
            rows.append({
                "definition": name, "gov_type": gt,
                "n_corridors": int(sel.sum()),
                "emission_Mt": round(gov.loc[sel, "E_e"].sum() / 1e6, 2),
                "emission_share_pct": round(gov.loc[sel, "E_e"].sum() / total_em * 100, 1),
            })
        rows.append({
            "definition": name, "gov_type": "RVI_strict",
            "n_corridors": int(is_vac_high.sum()),
            "emission_Mt": round(gov.loc[is_vac_high, "E_e"].sum() / 1e6, 2),
            "emission_share_pct": round(rvi_strict, 1),
        })
        rows.append({
            "definition": name, "gov_type": "RVI_any",
            "n_corridors": int((gtype == "Vacuum").sum()),
            "emission_Mt": round(gov.loc[gtype == "Vacuum", "E_e"].sum() / 1e6, 2),
            "emission_share_pct": round(rvi_any, 1),
        })
        logger.info(f"  [{name}] "
                    + ", ".join(f"{gt}={int((gtype == gt).sum())}" for gt in GOV_TYPES)
                    + f", RVI_any={rvi_any:.1f}%")

    # Corridors on which the two definitions disagree
    both = pressure & endpoint
    pressure_only = pressure & ~endpoint     # near-EU lanes w/o EU endpoint
    endpoint_only = ~pressure & endpoint     # EU-endpoint lanes below tau
    neither = ~pressure & ~endpoint
    diff = {
        "n_both": int(both.sum()),
        "emission_both_Mt": round(gov.loc[both, "E_e"].sum() / 1e6, 2),
        "n_pressure_only": int(pressure_only.sum()),
        "emission_pressure_only_Mt": round(gov.loc[pressure_only, "E_e"].sum() / 1e6, 2),
        "n_endpoint_only": int(endpoint_only.sum()),
        "emission_endpoint_only_Mt": round(gov.loc[endpoint_only, "E_e"].sum() / 1e6, 2),
        "n_neither": int(neither.sum()),
        "emission_neither_Mt": round(gov.loc[neither, "E_e"].sum() / 1e6, 2),
    }
    logger.info(f"  Definition disagreement: pressure-only={diff['n_pressure_only']} "
                f"({diff['emission_pressure_only_Mt']} Mt), "
                f"endpoint-only={diff['n_endpoint_only']} "
                f"({diff['emission_endpoint_only_Mt']} Mt)")

    out = pd.DataFrame(rows)
    out.to_csv(POLICY_DIR / f"robustness_fueleu_definition_{vt}.csv", index=False)
    summary = {
        "classification_table": rows,
        "definition_disagreement": diff,
        "note": ("Under the endpoint-based alternative the FuelEU-exposed category "
                 "collapses to zero in the mutually exclusive scheme, because its "
                 "scope (either endpoint in EU/EEA) coincides with the EU ETS "
                 "endpoint rule and the priority convention assigns those corridors "
                 "to EU_ETS. The pressure-field definition therefore supplies the "
                 "incremental 13.5%-of-emissions belt (near-EU lanes without an EU "
                 "endpoint) that the endpoint definition cannot see."),
    }
    with open(POLICY_DIR / f"robustness_fueleu_definition_{vt}.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Saved: robustness_fueleu_definition_{vt}.csv/.json")
    return summary


# ============================================================
# B) Range-constraint formulation equivalence
# ============================================================
def _solve_global_planner(corridors, ports, mell, budget_frac, fuel,
                          chain_mode=False, port_coords=None):
    """Global-planner MILP under either constraint formulation.

    chain_mode=False: aggregated counting form  z_e*k_e <= sum_{p in M_e} y_p
                      (the Module 05 implementation).
    chain_mode=True:  chain-partition form — selected intermediates must
                      contain a feasible bunkering chain (every sub-leg
                      within R_f); k_e=1 corridors reduce to the identical
                      aggregated constraint.
    """
    import pulp
    R_f = EMISSION_PARAMS["fuel_range_nm"].get(fuel, 5000)
    corr = m05.prepare_corridors(corridors, fuel)

    ports_f = ports[ports["is_feasible"]].copy()
    port_list = ports_f["port_code"].tolist()
    port_cost = ports_f.set_index("port_code")["cost_p"].to_dict()
    port_idx = {p: i for i, p in enumerate(port_list)}
    budget = sum(port_cost.values()) * budget_frac

    mell_f = mell[mell["fuel"] == fuel] if len(mell) else mell
    corr_inter = {}
    if len(mell_f) > 0:
        for cid, g in mell_f.groupby("corridor_id"):
            corr_inter[cid] = [p for p in g["intermediate_port"].tolist()
                               if p in port_idx]

    prob = pulp.LpProblem("GSC_robustness", pulp.LpMaximize)
    y = {p: pulp.LpVariable(f"y_{p}", cat="Binary") for p in port_list}
    z = {cid: pulp.LpVariable(f"z_{cid}", cat="Binary") for cid in corr["corridor_id"]}
    W = corr.set_index("corridor_id")["W_e"].to_dict()
    prob += pulp.lpSum(W[cid] * z[cid] for cid in z), "Total_Abatement"
    prob += (pulp.lpSum(port_cost[p] * y[p] for p in port_list) <= budget, "Budget")

    n_chains = 0
    n_chain_corridors = 0
    for _, row in corr.iterrows():
        cid = row["corridor_id"]
        o, d = row["origin_port"], row["dest_port"]
        if o not in port_idx or d not in port_idx:
            prob += z[cid] == 0, f"NoPort_{cid}"
            continue
        prob += z[cid] <= y[o], f"zYo_{cid}"
        prob += z[cid] <= y[d], f"zYd_{cid}"

        if not row["range_violating"]:
            continue
        k_e = max(1, int(row["k_e"]))
        inter = corr_inter.get(cid, [])
        if len(inter) < k_e:
            prob += z[cid] == 0, f"NoInter_{cid}"
            continue
        if not chain_mode or k_e == 1:
            # aggregated counting form; identical to the per-leg form for k_e=1
            prob += (z[cid] * k_e <= pulp.lpSum(y[p] for p in inter),
                     f"zInter_{cid}")
        else:
            # enumerate feasible chains through the corridor's candidates
            chains = _feasible_chains(o, d, inter, k_e, R_f, port_coords)
            if not chains:
                prob += z[cid] == 0, f"NoChain_{cid}"
                continue
            w = [pulp.LpVariable(f"w_{cid}_{j}", cat="Binary")
                 for j in range(len(chains))]
            prob += z[cid] <= pulp.lpSum(w), f"zChain_{cid}"
            for j, chain in enumerate(chains):
                for p in chain:
                    prob += w[j] <= y[p], f"w_{cid}_{j}_{p}"
            n_chains += len(chains)
            n_chain_corridors += 1

    try:
        solver = pulp.GUROBI(msg=0, timeLimit=MILP_PARAMS["time_limit_s"],
                             MIPGap=ROBUSTNESS_MIP_GAP)
        prob.solve(solver)
    except Exception:
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=MILP_PARAMS["time_limit_s"],
                                   gapRel=ROBUSTNESS_MIP_GAP)
        prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    if prob.status != 1:
        return {"status": status}
    selected = {p for p in port_list if y[p].varValue and y[p].varValue > 0.5}
    activated = {cid for cid in z if z[cid].varValue and z[cid].varValue > 0.5}
    return {"status": status,
            "objective_Mt": pulp.value(prob.objective) / 1e6,
            "n_ports": len(selected), "n_corridors": len(activated),
            "selected_ports": selected, "activated_corridors": activated,
            "n_chains": n_chains, "n_chain_corridors": n_chain_corridors}


def _feasible_chains(o, d, inter, k_e, R_f, port_coords):
    """All ordered k_e-tuples of distinct candidates such that every
    sub-leg (o->p1->...->pk->d) is within fuel range."""
    if port_coords is None or not inter:
        return []
    oc = port_coords[o]
    dc = port_coords[d]
    inter = [p for p in inter if p in port_coords]
    chains = []
    if k_e == 1:
        for p in inter:
            pc = port_coords[p]
            if (haversine_nm(oc[0], oc[1], pc[0], pc[1]) <= R_f
                    and haversine_nm(pc[0], pc[1], dc[0], dc[1]) <= R_f):
                chains.append((p,))
        return chains
    # k_e >= 2: DFS over permutations with pruning on the running sub-leg
    def extend(prefix, remaining, prev_port):
        if len(prefix) == k_e:
            pc = port_coords[prev_port]
            if haversine_nm(pc[0], pc[1], dc[0], dc[1]) <= R_f:
                chains.append(tuple(prefix))
            return
        prev_c = port_coords[prev_port]
        for p in remaining:
            pc = port_coords[p]
            if haversine_nm(prev_c[0], prev_c[1], pc[0], pc[1]) <= R_f:
                extend(prefix + [p],
                       [q for q in remaining if q != p], p)
    for p in inter:
        pc = port_coords[p]
        d0 = haversine_nm(oc[0], oc[1], pc[0], pc[1])
        if d0 <= R_f:
            extend([p], [q for q in inter if q != p], p)
    return chains


def run_range_constraint_robustness(vt):
    corridors = pd.read_parquet(PROCESSED_DIR / f"corridors_{vt}.parquet")
    ports = pd.read_parquet(PROCESSED_DIR / f"candidate_ports_{vt}.parquet")
    M_ell_df = pd.read_parquet(PROCESSED_DIR / f"M_ell_{vt}.parquet")
    ports = ports.drop_duplicates("port_code")
    mell = m05.prune_mell(M_ell_df, corridors, ports, K=MILP_PARAMS["m_ell_prune_k"])
    ports_g = pd.read_parquet(PROCESSED_DIR / "ports_global.parquet")
    port_coords = {r["port_code"]: (r["port_lon"], r["port_lat"])
                   for _, r in ports_g.drop_duplicates("port_code").iterrows()}

    # ---- k_e distribution (analytic part) ----
    k_stats = {}
    for fuel in [FUEL_REF, FUEL_SHORT]:
        rv = corridors[corridors[f"range_violating_{fuel}"]]
        kd = rv[f"k_e_{fuel}"].value_counts().sort_index()
        k_stats[fuel] = {
            "n_range_violating": int(len(rv)),
            "k_e_distribution": {int(k): int(v) for k, v in kd.items()},
            "share_k_e_1": round(float((rv[f"k_e_{fuel}"] == 1).mean()) * 100, 1),
        }
        logger.info(f"  [{fuel}] range-violating={len(rv)}, "
                    f"k_e=1 share={k_stats[fuel]['share_k_e_1']}%")

    # ---- budget pool verification ----
    feas = ports[ports["is_feasible"]]
    pool = feas["cost_p"].sum()
    pool_check = {
        "n_feasible_ports": int(len(feas)),
        "full_upgrade_pool_MUSD_yr": round(float(pool), 1),
        "budget20_MUSD_yr": round(float(pool) * 0.20, 1),
        "budget10_MUSD_yr": round(float(pool) * 0.10, 1),
        "mean_cost_MUSD_yr": round(float(feas["cost_p"].mean()), 2),
    }
    logger.info(f"  Cost pool: {pool_check['full_upgrade_pool_MUSD_yr']} M USD/yr "
                f"({pool_check['n_feasible_ports']} feasible ports); "
                f"20% = {pool_check['budget20_MUSD_yr']}, "
                f"10% = {pool_check['budget10_MUSD_yr']}")

    # ---- numerical equivalence runs ----
    rows = []
    for fuel, label in [(FUEL_REF, "methanol"), (FUEL_SHORT, "ammonia")]:
        for frac in BUDGET_FRACS:
            res_agg = _solve_global_planner(corridors, ports, mell, frac, fuel,
                                            chain_mode=False)
            res_chain = _solve_global_planner(corridors, ports, mell, frac, fuel,
                                               chain_mode=True, port_coords=port_coords)
            ok = (res_agg.get("status") == "Optimal"
                  and res_chain.get("status") == "Optimal")
            obj_a = res_agg.get("objective_Mt") or 0.0
            obj_c = res_chain.get("objective_Mt") or 0.0
            obj_gap = (abs(obj_a - obj_c) / max(obj_a, 1e-9) * 100) if ok else float("nan")
            pa = res_agg.get("selected_ports") or set()
            pc = res_chain.get("selected_ports") or set()
            ca = res_agg.get("activated_corridors") or set()
            cc = res_chain.get("activated_corridors") or set()
            same_ports = ok and (pa == pc)
            same_corr = ok and (ca == cc)
            jaccard = (len(pa & pc) / max(1, len(pa | pc))) if ok else float("nan")
            rows.append({
                "fuel": label, "budget_fraction": frac,
                "obj_aggregated_Mt": round(obj_a, 3) if ok else None,
                "obj_chain_Mt": round(obj_c, 3) if ok else None,
                "obj_gap_pct": round(obj_gap, 4) if ok else None,
                "n_ports_aggregated": res_agg.get("n_ports"),
                "n_ports_chain": res_chain.get("n_ports"),
                "port_set_jaccard": round(jaccard, 4) if ok else None,
                "same_port_set": bool(same_ports),
                "same_activated_set": bool(same_corr),
                "n_chain_corridors": res_chain.get("n_chain_corridors", 0),
                "n_chains_enumerated": res_chain.get("n_chains", 0),
                "status_agg": res_agg.get("status"),
                "status_chain": res_chain.get("status"),
            })
            logger.info(f"  [{label}, {frac:.0%}] agg={obj_a:.2f} Mt "
                        f"({res_agg.get('n_ports')} ports) vs "
                        f"chain={obj_c:.2f} Mt "
                        f"({res_chain.get('n_ports')} ports); "
                        f"gap={obj_gap:.3f}%, same ports={same_ports}")

    out = pd.DataFrame(rows)
    out.to_csv(POLICY_DIR / f"robustness_range_constraint_{vt}.csv", index=False)
    summary = {
        "k_e_distribution": k_stats,
        "budget_pool_check": pool_check,
        "runs": rows,
        "note": ("For the methanol reference scenario every range-violating "
                 "corridor has k_e = 1, where the aggregated form z_e*k_e <= "
                 "sum(y_p) is algebraically identical to the per-leg form "
                 "(z_e*1 = z_e), so the two formulations coincide by construction. "
                 "For ammonia (R_f = 5,000 nm) corridors with k_e >= 2 exist; the "
                 "chain-partition formulation (feasible bunkering chain required) "
                 "is the tightest per-leg implementation, so the aggregated-vs-chain "
                 "comparison brackets any slack of the aggregated form."),
    }
    with open(POLICY_DIR / f"robustness_range_constraint_{vt}.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Saved: robustness_range_constraint_{vt}.csv/.json")
    return summary


# ============================================================
# Main
# ============================================================
def main(vt):
    logger.info("=" * 60)
    logger.info(f"ROBUSTNESS CHECKS — v4 ({vt})")
    logger.info("=" * 60)

    logger.info("\n[A] FuelEU definition robustness (pressure field vs endpoint)")
    fueleu_summary = run_fueleu_definition_robustness(vt)

    logger.info("\n[B] Range-constraint formulation equivalence")
    range_summary = run_range_constraint_robustness(vt)

    combined = {"version": "v4-robustness", "vessel_type": vt,
                "fueleu_definition": fueleu_summary,
                "range_constraint": range_summary}
    with open(POLICY_DIR / f"robustness_summary_{vt}.json", "w") as f:
        json.dump(combined, f, indent=2)
    logger.info(f"\nSaved: robustness_summary_{vt}.json")
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vessel-type", default=VESSEL_TYPE)
    args = parser.parse_args()
    main(args.vessel_type)
