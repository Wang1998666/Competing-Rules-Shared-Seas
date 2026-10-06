"""
Module 17 (v4): Supplementary Sensitivity Analyses
===================================================
Four additional sensitivity/robustness experiments covering the data- and
definition-level assumptions that the manuscript forward-references:

  A. Berth-mask sensitivity (Eq. 18 feasibility screen):
     no mask (all 1,051 candidates) / baseline (944) / berths>=1 (539) /
     berths>=3 (305). Re-solves the global planner (methanol, aggregated
     range form) at 10% and 20% budgets under each screen.

  B. Distance-measurement sensitivity (detour / sailed-distance error):
     corridor mean sailed distance and M_ell sub-leg distances scaled by
     {0.95, 1.00, 1.05}; range-violating flags recomputed from the scaled
     distances, M_ell rows re-filtered by the scaled sub-leg condition.
     Brackets the regionalized detour factors (1.02-1.05) of the fallback
     branch. Targets the range-constraint channel only; the cost channel
     is covered by the uniform-cost test of Module 13.

  C. Corridor frequency threshold n_min:
     n_min in {4 (reference), 8, 16}. Reports corridor counts, emission
     coverage, governance shares, RVI, and re-solves the global planner
     at the 20% budget. (Downward relaxation n_min < 4 is bounded by the
     95.1% leg-level coverage of the reference set, Table 1.)

  D. Governance priority order:
     mutually exclusive attribution under the reference order
     EU > BRI > FuelEU and three alternatives. Reports four-type shares
     and RVI; verifies that the vacuum set (RE_e = 0) and the rule-sphere
     MILP domains (EU scope: w_e > 0; BRI scope: membership flag) are
     invariant by construction.

Outputs (02_数据_output/policy_analysis_v4/, NEW files only):
  sensitivity_berth_mask_{vt}.csv / .json
  sensitivity_distance_{vt}.csv / .json
  sensitivity_nmin_{vt}.csv / .json
  sensitivity_priority_order_{vt}.csv / .json
  sensitivity_supp_summary_{vt}.json

Run:  python 17_sensitivity_supp.py [--vessel-type container]
"""

import sys
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (PROCESSED_DIR, POLICY_DIR, VESSEL_TYPE, MILP_PARAMS)
from importlib import import_module
m05 = import_module("05_milp_solving")
m16 = import_module("16_robustness_checks")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FUEL_REF = "green_methanol"
R_F = 8000.0
BUDGET_FRACS = [0.10, 0.20]
GOV_TYPES = ["EU_ETS", "BRI_only", "FuelEU_exposed", "Vacuum"]


# ============================================================
# Shared: solve + comparison helper
# ============================================================
def solve_and_compare(corridors, ports, mell, frac, baseline_ports=None):
    res = m16._solve_global_planner(corridors, ports, mell, frac, FUEL_REF,
                                    chain_mode=False)
    if res.get("status") != "Optimal":
        return {"status": res.get("status")}
    out = {"status": "Optimal",
           "objective_Mt": round(res["objective_Mt"], 2),
           "n_ports": res["n_ports"],
           "n_corridors": res["n_corridors"]}
    if baseline_ports is not None:
        sel = res["selected_ports"]
        union = len(sel | baseline_ports)
        out["port_jaccard_vs_ref"] = round(len(sel & baseline_ports) / union, 3)
    return out


def load_inputs(vt):
    corridors = pd.read_parquet(PROCESSED_DIR / f"corridors_{vt}.parquet")
    ports = pd.read_parquet(PROCESSED_DIR / f"candidate_ports_{vt}.parquet")
    M_ell_df = pd.read_parquet(PROCESSED_DIR / f"M_ell_{vt}.parquet")
    ports = ports.drop_duplicates("port_code")
    mell = m05.prune_mell(M_ell_df, corridors, ports, K=MILP_PARAMS["m_ell_prune_k"])
    return corridors, ports, mell


# ============================================================
# A) Berth-mask sensitivity
# ============================================================
def run_berth_mask(vt, corridors, ports, mell):
    berths = ports["n_container_berths"].fillna(0)
    variants = [
        ("baseline_eq18", ports["is_feasible"]),
        ("no_mask", pd.Series(True, index=ports.index)),
        ("berths_ge1", berths >= 1),
        ("berths_ge3", berths >= 3),
    ]

    # reference port sets from the baseline screen, one per budget
    ref = {}
    obj_ref = {}
    for frac in BUDGET_FRACS:
        r = m16._solve_global_planner(corridors, ports, mell, frac,
                                      FUEL_REF, chain_mode=False)
        ref[frac] = r["selected_ports"]
        obj_ref[frac] = r["objective_Mt"]
    logger.info(f"  [baseline_eq18] feasible={int(ports['is_feasible'].sum())}, "
                f"obj10={obj_ref[0.10]:.2f} Mt, obj20={obj_ref[0.20]:.2f} Mt (reference)")

    rows = []
    for name, mask in variants:
        ports_v = ports.copy()
        ports_v["is_feasible"] = mask
        feas = ports_v[ports_v["is_feasible"]]
        row = {"mask": name, "n_feasible_ports": int(mask.sum()),
               "cost_pool_MUSD_yr": round(float(feas["cost_p"].sum()), 1)}
        for frac in BUDGET_FRACS:
            res = solve_and_compare(corridors, ports_v, mell, frac,
                                    baseline_ports=ref[frac])
            tag = f"{int(frac*100)}pct"
            if res.get("status") == "Optimal":
                row[f"obj_{tag}_Mt"] = res["objective_Mt"]
                row[f"ports_{tag}"] = res["n_ports"]
                row[f"corr_{tag}"] = res["n_corridors"]
                row[f"jaccard_{tag}"] = res["port_jaccard_vs_ref"]
            else:
                row[f"obj_{tag}_Mt"] = None
        rows.append(row)
        logger.info(f"  [{name}] feasible={row['n_feasible_ports']}, "
                    f"pool={row['cost_pool_MUSD_yr']} M, "
                    f"obj20={row.get('obj_20pct_Mt')} Mt, "
                    f"ports20={row.get('ports_20pct')}, "
                    f"jac20={row.get('jaccard_20pct')}")

    out = pd.DataFrame(rows)
    out.to_csv(POLICY_DIR / f"sensitivity_berth_mask_{vt}.csv", index=False)
    summary = {"variants": rows,
               "note": ("Baseline is the Eq. 18 screen (container berth OR "
                        ">=24 annual calls fallback). berths_ge1 drops the "
                        "calls-based fallback; berths_ge3 is a stricter "
                        "physical screen; no_mask admits all candidates. "
                        "The budget is a fixed fraction of each variant's "
                        "own cost pool, so the comparison isolates the "
                        "composition of the feasible set, not its size.")}
    with open(POLICY_DIR / f"sensitivity_berth_mask_{vt}.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Saved: sensitivity_berth_mask_{vt}.csv/.json")
    return summary


# ============================================================
# B) Distance-measurement sensitivity
# ============================================================
def run_distance(vt, corridors, ports, mell):
    rows = []
    for scale in [0.95, 1.00, 1.05]:
        corr_v = corridors.copy()
        corr_v["mean_distance"] = corridors["mean_distance"] * scale
        mell_v = mell.copy()
        if scale != 1.0:
            keep = ((mell["d_origin_nm"] * scale <= R_F)
                    & (mell["d_dest_nm"] * scale <= R_F))
            mell_v = mell[keep]
        rv = (corr_v["mean_distance"] > R_F)
        row = {"scale": scale,
               "n_range_violating": int(rv.sum()),
               "n_mell_rows": int(len(mell_v))}
        for frac in BUDGET_FRACS:
            res = solve_and_compare(corridors, ports, mell_v, frac)
            tag = f"{int(frac*100)}pct"
            if res.get("status") == "Optimal":
                row[f"obj_{tag}_Mt"] = res["objective_Mt"]
                row[f"ports_{tag}"] = res["n_ports"]
                row[f"corr_{tag}"] = res["n_corridors"]
            else:
                row[f"obj_{tag}_Mt"] = None
        rows.append(row)
        logger.info(f"  [scale={scale}] rv={row['n_range_violating']}, "
                    f"mell rows={row['n_mell_rows']}, "
                    f"obj20={row.get('obj_20pct_Mt')} Mt")

    out = pd.DataFrame(rows)
    out.to_csv(POLICY_DIR / f"sensitivity_distance_{vt}.csv", index=False)
    summary = {"runs": rows,
               "note": ("Distances enter the siting model only through the "
                        "range constraint (corridor distance vs R_f and the "
                        "M_ell sub-leg condition). The +/-5% bracket covers "
                        "the regionalized detour factors (1.02-1.05) of the "
                        "great-circle fallback and typical sailed-distance "
                        "measurement error. Port costs (remoteness premium) "
                        "are held fixed; the cost channel is covered by the "
                        "uniform-cost sensitivity of Module 13.")}
    with open(POLICY_DIR / f"sensitivity_distance_{vt}.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Saved: sensitivity_distance_{vt}.csv/.json")
    return summary


# ============================================================
# C) Corridor frequency threshold n_min
# ============================================================
def run_nmin(vt, corridors, ports, mell):
    gov = pd.read_parquet(POLICY_DIR / f"governance_corridors_{vt}.parquet")
    gov = gov.merge(corridors[["corridor_id", "frequency"]], on="corridor_id")
    total_all = gov["E_e"].sum()
    p75 = gov.loc[gov["E_e"] > 0, "E_e"].quantile(0.75)

    rows = []
    for nmin in [4, 8, 16]:
        sel = gov[gov["frequency"] >= nmin]
        total = sel["E_e"].sum()
        row = {"n_min": nmin, "n_corridors": int(len(sel)),
               "emission_Mt": round(total / 1e6, 1),
               "emission_coverage_pct": round(total / total_all * 100, 1)}
        for gt in GOV_TYPES:
            m = sel["gov_type"] == gt
            row[f"{gt}_share_pct"] = round(sel.loc[m, "E_e"].sum() / total * 100, 1)
        vac = sel["gov_type"] == "Vacuum"
        row["RVI_any_pct"] = round(sel.loc[vac, "E_e"].sum() / total * 100, 1)
        row["RVI_strict_pct"] = round(
            sel.loc[vac & (sel["E_e"] >= p75), "E_e"].sum() / total * 100, 1)
        rows.append(row)
        logger.info(f"  [n_min={nmin}] corridors={row['n_corridors']}, "
                    f"coverage={row['emission_coverage_pct']}%, "
                    f"RVI_any={row['RVI_any_pct']}%")

    # MILP re-solves at the 20% budget
    for nmin in [8, 16]:
        corr_v = corridors[corridors["frequency"] >= nmin]
        mell_v = mell[mell["corridor_id"].isin(corr_v["corridor_id"])]
        res = solve_and_compare(corr_v, ports, mell_v, 0.20)
        for r in rows:
            if r["n_min"] == nmin and res.get("status") == "Optimal":
                r["obj_20pct_Mt"] = res["objective_Mt"]
                r["ports_20pct"] = res["n_ports"]
                r["corr_20pct"] = res["n_corridors"]
        logger.info(f"  [n_min={nmin}] MILP 20%: "
                    f"{res.get('objective_Mt')} Mt, {res.get('n_ports')} ports")

    out = pd.DataFrame(rows)
    out.to_csv(POLICY_DIR / f"sensitivity_nmin_{vt}.csv", index=False)
    summary = {"runs": rows,
               "note": ("Downward relaxation (n_min < 4) is bounded by Table 1: "
                        "the reference set already carries 95.1% of leg-level "
                        "emissions, so admitting sporadic lanes can shift "
                        "aggregate quantities by at most the residual 4.9%.")}
    with open(POLICY_DIR / f"sensitivity_nmin_{vt}.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Saved: sensitivity_nmin_{vt}.csv/.json")
    return summary


# ============================================================
# D) Governance priority order
# ============================================================
def _assign_priority(ets, bri, fueleu, order):
    """Mutually exclusive type under a given priority order."""
    flags = {"EU_ETS": ets, "BRI_only": bri, "FuelEU_exposed": fueleu}
    n = len(ets)
    out = np.array(["Vacuum"] * n, dtype=object)
    for name in order:            # first match wins
        out[flags[name].to_numpy() & (out == "Vacuum")] = name
    return pd.Series(out, index=ets.index)


def run_priority_order(vt):
    gov = pd.read_parquet(POLICY_DIR / f"governance_corridors_{vt}.parquet")
    total = gov["E_e"].sum()
    p75 = gov.loc[gov["E_e"] > 0, "E_e"].quantile(0.75)
    ets, bri, fueleu = gov["eu_ets_endpoint"], gov["is_bri"], gov["fueleu_exposed"]

    orders = [
        ("EU>BRI>FuelEU (ref)", ["EU_ETS", "BRI_only", "FuelEU_exposed"]),
        ("BRI>EU>FuelEU", ["BRI_only", "EU_ETS", "FuelEU_exposed"]),
        ("EU>FuelEU>BRI", ["EU_ETS", "FuelEU_exposed", "BRI_only"]),
        ("FuelEU>EU>BRI", ["FuelEU_exposed", "EU_ETS", "BRI_only"]),
    ]
    rows = []
    vac_sets = {}
    for name, order in orders:
        gtype = _assign_priority(ets, bri, fueleu, order)
        vac = gtype == "Vacuum"
        vac_sets[name] = set(gov.loc[vac, "corridor_id"])
        row = {"order": name}
        for gt in GOV_TYPES:
            m = gtype == gt
            row[f"{gt}_n"] = int(m.sum())
            row[f"{gt}_share_pct"] = round(gov.loc[m, "E_e"].sum() / total * 100, 1)
        row["RVI_any_pct"] = round(gov.loc[vac, "E_e"].sum() / total * 100, 1)
        row["RVI_strict_pct"] = round(
            gov.loc[vac & (gov["E_e"] >= p75), "E_e"].sum() / total * 100, 1)
        rows.append(row)
        logger.info(f"  [{name}] " + ", ".join(
            f"{gt}={row[f'{gt}_share_pct']}%" for gt in GOV_TYPES)
            + f", RVI_any={row['RVI_any_pct']}%")

    ref_vac = vac_sets[orders[0][0]]
    vac_invariant = all(s == ref_vac for s in vac_sets.values())
    out = pd.DataFrame(rows)
    out.to_csv(POLICY_DIR / f"sensitivity_priority_order_{vt}.csv", index=False)
    summary = {"runs": rows,
               "vacuum_set_invariant": vac_invariant,
               "note": ("The vacuum set (RE_e = 0) is order-invariant by "
                        "construction, so the RVI is identical under all "
                        "orders. The rule-sphere MILP domains are likewise "
                        "unaffected: the EU scope uses the ETS exposure "
                        "weight (w_e > 0) and the BRI scope uses membership "
                        "flags, neither of which involves the mutually "
                        "exclusive types. Only the descriptive split of the "
                        "overlapping corridors (1,014 BRI-and-ETS lanes) "
                        "moves between the EU_ETS and BRI_only labels.")}
    with open(POLICY_DIR / f"sensitivity_priority_order_{vt}.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Saved: sensitivity_priority_order_{vt}.csv/.json "
                f"(vacuum invariant: {vac_invariant})")
    return summary


# ============================================================
# Main
# ============================================================
def main(vt):
    logger.info("=" * 60)
    logger.info(f"SUPPLEMENTARY SENSITIVITY — v4 ({vt})")
    logger.info("=" * 60)

    corridors, ports, mell = load_inputs(vt)

    logger.info("\n[A] Berth-mask sensitivity")
    berth_summary = run_berth_mask(vt, corridors, ports, mell)

    logger.info("\n[B] Distance-measurement sensitivity (+/-5%)")
    dist_summary = run_distance(vt, corridors, ports, mell)

    logger.info("\n[C] Corridor frequency threshold n_min")
    nmin_summary = run_nmin(vt, corridors, ports, mell)

    logger.info("\n[D] Governance priority order")
    prio_summary = run_priority_order(vt)

    combined = {"version": "v4-sensitivity-supp", "vessel_type": vt,
                "berth_mask": berth_summary,
                "distance": dist_summary,
                "n_min": nmin_summary,
                "priority_order": prio_summary}
    with open(POLICY_DIR / f"sensitivity_supp_summary_{vt}.json", "w") as f:
        json.dump(combined, f, indent=2)
    logger.info(f"\nSaved: sensitivity_supp_summary_{vt}.json")
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vessel-type", default=VESSEL_TYPE)
    args = parser.parse_args()
    main(args.vessel_type)
