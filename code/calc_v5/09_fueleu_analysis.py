"""
Module 09 (v2): FuelEU Compliance Pressure Field & Rule-Competition MILP
========================================================================
Builds the FuelEU pressure field P(h) = max_{p∈EU} exp(−d(h,p)/λ), λ=2000 nm
(Eq. 19), and solves the rule-competition MILP family:

  * view='global': objective = global abatement (M^G with FuelEU constraints)
  * view='eu'    : objective = EU-related corridor abatement (M^EU)

v2 fixes vs v1:
  * ETS revenue recycling in CLEAN units: c_p is in M USD/yr, E_e in t/yr,
    so the recycled budget ΔB = γ·P_ETS·Σ_e w_e·E_e·z_e is added directly
    (no artificial 1e8 scaling).
  * W_e recomputed per fuel scenario (methanol reference).
  * Feasibility mask applied (Eq. 18).
  * Top-N EU ports forced open via MILP_PARAMS["fueleu_forced_ports"].

Outputs (02_数据_output/policy_analysis_v2/): fueleu_* files with the same
suffix convention as v1.
"""

import sys
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import h3

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PROCESSED_DIR, H3_DIR, POLICY_DIR, EMISSION_PARAMS, MILP_PARAMS,
    VESSEL_TYPE, get_data_paths, OUTPUT_DIR,
)

from importlib import import_module
_03c = import_module("03c_h3_policy_coverage")
EU_EEA_COUNTRIES = _03c.EU_EEA_COUNTRIES  # unified country set

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DECAY_LAMBDA_NM = 2000.0
FUELEU_PRESSURE_TAU = 0.30
FUELEU_TAU_SENSITIVITY = [0.10, 0.30, 0.50]


def haversine_nm(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 3440.065 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_fueleu_pressure_field(vessel_type: str):
    logger.info(f"Building FuelEU pressure field v2: {vessel_type}")
    # Use v1 dense H3 grid (861K cells) for the pressure field so that
    # corridor paths (v1-grid based) and the pressure raster share the same
    # cell IDs — keeps Fig 11 continuous instead of sparse v2 cells.
    h3_file = OUTPUT_DIR / "h3_grid" / f"h3_emission_grid_{vessel_type}.parquet"
    ports_file = PROCESSED_DIR / "ports_global.parquet"
    if not h3_file.exists():
        raise RuntimeError("H3 grid not found — run 03 first")
    h3_df = pd.read_parquet(h3_file)
    ports = pd.read_parquet(ports_file)
    eu_ports = ports.drop_duplicates("port_code")
    eu_ports = eu_ports[eu_ports["country_code"].isin(EU_EEA_COUNTRIES) &
                        eu_ports["port_lon"].notna() & eu_ports["port_lat"].notna()]
    logger.info(f"  EU/EEA ports: {len(eu_ports)}")

    h3_lats = h3_df["lat"].values
    h3_lons = h3_df["lon"].values
    eu_lats = eu_ports["port_lat"].values
    eu_lons = eu_ports["port_lon"].values

    chunk_size = 50000
    pressure = np.zeros(len(h3_df))
    for start in range(0, len(h3_df), chunk_size):
        end = min(start + chunk_size, len(h3_df))
        lat1 = np.radians(h3_lats[start:end])[:, None]
        lon1 = np.radians(h3_lons[start:end])[:, None]
        lat2 = np.radians(eu_lats)[None, :]
        lon2 = np.radians(eu_lons)[None, :]
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        dist_nm = 2 * 3440.065 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        pressure[start:end] = np.exp(-dist_nm / DECAY_LAMBDA_NM).max(axis=1)

    h3_df["fueleu_pressure"] = pressure
    h3_df.to_parquet(POLICY_DIR / f"fueleu_pressure_{vessel_type}.parquet", index=False)
    logger.info(f"  Pressure stats: mean={pressure.mean():.4f}, max={pressure.max():.4f}, "
                f"cells P>0.1: {(pressure > 0.1).sum():,}")
    logger.info("  Saved: fueleu_pressure (v2)")
    return h3_df


def _suffix(view, gamma, force_fueleu, budget_fraction):
    force_tag = "" if force_fueleu else "_noforce"
    pct = int(round(budget_fraction * 100))
    if view != "global" or gamma != 0.0 or not force_fueleu or pct != 20:
        return f"_view{view}_b{pct}_g{gamma:.1f}{force_tag}"
    return ""


def run_fueleu_constrained_milp(vessel_type: str, budget_fraction: float = 0.20,
                                gamma: float = 0.0, ets_price: float = 80.0,
                                view: str = "global", force_fueleu: bool = True,
                                fuel: str = "green_methanol"):
    logger.info(f"  FuelEU/ETS MILP v2: view={view}, B={budget_fraction:.0%}, "
                f"gamma={gamma}, P_ETS={ets_price}")

    from importlib import import_module
    m05 = import_module("05_milp_solving")

    corridors = pd.read_parquet(PROCESSED_DIR / f"corridors_{vessel_type}.parquet")
    ports_dedup = pd.read_parquet(PROCESSED_DIR / f"candidate_ports_{vessel_type}.parquet")
    ports_dedup = ports_dedup.drop_duplicates("port_code")

    mell_raw = pd.read_parquet(PROCESSED_DIR / f"M_ell_{vessel_type}.parquet")
    mell = m05.prune_mell(mell_raw, corridors, ports_dedup,
                          K=MILP_PARAMS["m_ell_prune_k"])

    # feasible ports only (Eq. 18)
    ports_f = ports_dedup[ports_dedup["is_feasible"]].copy()
    port_list = ports_f["port_code"].tolist()
    cost_p = ports_f.set_index("port_code")["cost_p"]
    port_idx = {p: i for i, p in enumerate(port_list)}
    n_ports = len(port_list)
    total_cost = cost_p.sum()
    logger.info(f"  Feasible ports: {n_ports}; cost pool: {total_cost:.1f} M USD/yr")

    # EU ports forced open (top-N by corridor traffic); N is budget-aware:
    # the forced set must remain affordable under the base budget, otherwise
    # the model is infeasible at tight budgets (v2 fix).
    eu_port_codes = set(ports_f[ports_f["country_code"].isin(EU_EEA_COUNTRIES)]["port_code"])
    corr_eu_count = pd.concat([
        corridors.loc[corridors["origin_port"].isin(eu_port_codes), "origin_port"],
        corridors.loc[corridors["dest_port"].isin(eu_port_codes), "dest_port"],
    ]).value_counts()
    base_budget = total_cost * budget_fraction
    ranked = [c for c in corr_eu_count.index if c in port_idx]
    if ranked:
        cum_cost = np.cumsum([cost_p[p] for p in ranked])
        n_affordable = int((cum_cost <= base_budget * 0.8).sum())
    else:
        n_affordable = 0
    n_force = min(MILP_PARAMS["fueleu_forced_ports"], len(ranked), max(n_affordable, 1))
    eu_forced_ports = set(ranked[:n_force])
    logger.info(f"  EU ports forced (top-{n_force}, budget-aware): "
                f"{len(eu_forced_ports)}; affordable cap was {n_affordable}")

    # corridor data with fuel-specific W_e (v2: no stale cross-scenario W_e)
    corr = m05.prepare_corridors(corridors, fuel)
    corr_list = corr[["corridor_id", "origin_port", "dest_port", "E_e", "W_e",
                      "mean_distance", "k_e", "range_violating"]].copy()
    corr_list = corr_list[corr_list["origin_port"].isin(port_idx) &
                          corr_list["dest_port"].isin(port_idx)]

    # EU ETS exposure weight (rule-consistent endpoint definition)
    port_eu = ports_f.set_index("port_code")["country_code"].isin(EU_EEA_COUNTRIES)
    corr_list["origin_eu"] = corr_list["origin_port"].map(port_eu).fillna(False)
    corr_list["dest_eu"] = corr_list["dest_port"].map(port_eu).fillna(False)
    corr_list["eu_ets_weight"] = np.where(
        corr_list["origin_eu"] & corr_list["dest_eu"], 1.0,
        np.where(corr_list["origin_eu"] | corr_list["dest_eu"], 0.5, 0.0))
    corr_list["eu_related"] = corr_list["eu_ets_weight"] > 0

    mell_dict = {}
    if len(mell) > 0:
        mell_f = mell[mell["fuel"] == fuel]
        for cid, g in mell_f.groupby("corridor_id"):
            mell_dict[cid] = [p for p in g["intermediate_port"].tolist() if p in port_idx]

    try:
        import pulp
    except ImportError:
        logger.error("pulp required for rule-competition MILP v2")
        return None

    prob = pulp.LpProblem(f"rule_MILP_v2_view{view}", pulp.LpMaximize)

    y = {p: pulp.LpVariable(f"y_{p}", cat="Binary") for p in port_list}
    z = {cid: pulp.LpVariable(f"z_{cid}", cat="Binary")
         for cid in corr_list["corridor_id"]}

    if view == "eu":
        obj = pulp.lpSum(row["W_e"] * z[row["corridor_id"]]
                         for _, row in corr_list.iterrows() if row["eu_related"])
    else:
        obj = pulp.lpSum(row["W_e"] * z[row["corridor_id"]]
                         for _, row in corr_list.iterrows())
    prob += obj

    # Budget with ETS revenue recycling (Eq. 25) — clean units (M USD/yr)
    base_budget = total_cost * budget_fraction
    prob += (pulp.lpSum(cost_p[p] * y[p] for p in port_list) - pulp.lpSum(
        gamma * ets_price * row["eu_ets_weight"] * row["E_e"] / 1e6 * z[row["corridor_id"]]
        for _, row in corr_list.iterrows())
        <= base_budget, "budget_ets_recycling")

    if force_fueleu:
        for p in eu_forced_ports:
            prob += (y[p] == 1, f"fueleu_{p}")

    # endpoint (Eq. 14) and intermediate (Eq. 15) constraints
    for _, row in corr_list.iterrows():
        cid, o, d = row["corridor_id"], row["origin_port"], row["dest_port"]
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

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=MILP_PARAMS["time_limit_s"],
                               gapRel=MILP_PARAMS["mip_gap"])
    prob.solve(solver)
    if prob.status != pulp.LpStatusOptimal:
        logger.warning(f"  MILP status {pulp.LpStatus[prob.status]} — skipping")
        return None

    selected = [p for p in port_list if y[p].varValue and y[p].varValue > 0.5]
    activated = [cid for cid in corr_list["corridor_id"]
                 if z[cid].varValue and z[cid].varValue > 0.5]
    obj_view = pulp.value(prob.objective)
    obj_global = corr_list.loc[corr_list["corridor_id"].isin(activated), "W_e"].sum()

    # ETS revenue realized (M USD/yr)
    ets_revenue = corr_list.loc[corr_list["corridor_id"].isin(activated),
                                ["eu_ets_weight", "E_e"]].eval("eu_ets_weight * E_e").sum() \
        * gamma * ets_price / 1e6

    # Baseline: same-scope no-force run of this module
    bl_json = POLICY_DIR / f"fueleu_comparison_{vessel_type}{_suffix(view, 0.0, False, budget_fraction)}.json"
    baseline_global = baseline_view = None
    baseline_ports = baseline_corr = set()
    if bl_json.exists():
        bl = json.load(open(bl_json))
        baseline_global = bl["rule_obj_global_Mt"] * 1e6
        baseline_view = bl["rule_obj_view_Mt"] * 1e6
        f_act = POLICY_DIR / f"fueleu_activated_corridors_{vessel_type}{_suffix(view, 0.0, False, budget_fraction)}.csv"
        f_sel = POLICY_DIR / f"fueleu_selected_ports_{vessel_type}{_suffix(view, 0.0, False, budget_fraction)}.csv"
        if f_act.exists():
            baseline_corr = set(pd.read_csv(f_act)["corridor_id"])
        if f_sel.exists():
            baseline_ports = set(pd.read_csv(f_sel)["port_code"])
    else:
        from config import RESULTS_DIR
        sel_file = RESULTS_DIR / f"selected_ports_{vessel_type}.csv"
        act_file = RESULTS_DIR / f"activated_corridors_{vessel_type}.csv"
        if sel_file.exists():
            baseline_ports = set(pd.read_csv(sel_file)["port_code"])
        if act_file.exists():
            baseline_corr = set(pd.read_csv(act_file)["corridor_id"])
        baseline_global = corr_list.loc[corr_list["corridor_id"].isin(baseline_corr), "W_e"].sum()
        baseline_view = corr_list.loc[corr_list["corridor_id"].isin(baseline_corr), "W_e"].sum()
        logger.warning("  No no-force baseline found — using Module 05 outputs as baseline")

    mpc = (baseline_global - obj_global) / baseline_global * 100.0 if baseline_global and baseline_global > 0 else float("nan")

    comparison = {
        "version": "v2", "view": view, "budget_fraction": budget_fraction,
        "gamma": gamma, "ets_price_usd": ets_price, "fuel": fuel,
        "base_budget_MUSD_yr": round(base_budget, 1),
        "ets_revenue_MUSD_yr": round(ets_revenue, 2),
        "baseline_obj_global_Mt": round(baseline_global / 1e6, 2) if baseline_global else None,
        "baseline_obj_view_Mt": round(baseline_view / 1e6, 2) if baseline_view else None,
        "rule_obj_view_Mt": round(obj_view / 1e6, 2),
        "rule_obj_global_Mt": round(obj_global / 1e6, 2),
        "MPC_global_pct": round(mpc, 2) if mpc == mpc else None,
        "baseline_n_ports": len(baseline_ports),
        "rule_n_ports": len(selected),
        "eu_ports_forced": len(eu_forced_ports),
        "new_ports_added": len(set(selected) - baseline_ports),
        "ports_removed": len(baseline_ports - set(selected)),
        "new_corridors_activated": len(set(activated) - baseline_corr),
        "corridors_lost": len(baseline_corr - set(activated)),
    }
    logger.info(f"  Objective: {comparison['rule_obj_view_Mt']} Mt (view), "
                f"{comparison['rule_obj_global_Mt']} Mt (global), MPC={comparison['MPC_global_pct']}%")

    suffix = _suffix(view, gamma, force_fueleu, budget_fraction)
    pd.DataFrame({"port_code": selected}).to_csv(
        POLICY_DIR / f"fueleu_selected_ports_{vessel_type}{suffix}.csv", index=False)
    pd.DataFrame({"corridor_id": activated}).to_csv(
        POLICY_DIR / f"fueleu_activated_corridors_{vessel_type}{suffix}.csv", index=False)
    with open(POLICY_DIR / f"fueleu_comparison_{vessel_type}{suffix}.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # v2: port-change detail (added vs removed vs baseline) for Fig 11
    new_ports = set(selected) - baseline_ports
    removed_ports = baseline_ports - set(selected)
    port_detail = ports_f[ports_f["port_code"].isin(new_ports | removed_ports)].copy()
    port_detail["change"] = port_detail["port_code"].apply(
        lambda p: "added" if p in new_ports else "removed")
    port_detail.to_csv(POLICY_DIR / f"fueleu_port_changes_{vessel_type}{suffix}.csv",
                       index=False)
    logger.info(f"  Saved: fueleu_*{suffix} (v2)")
    return comparison


def main(vessel_type: str):
    logger.info("=" * 60)
    logger.info(f"FUEL-EU / RULE-COMPETITION MILP — v2 ({vessel_type})")
    logger.info("=" * 60)
    pressure_file = POLICY_DIR / f"fueleu_pressure_{vessel_type}.parquet"
    if pressure_file.exists():
        logger.info("  Pressure field exists — skipping rebuild")
    else:
        build_fueleu_pressure_field(vessel_type)

    # Key experiment grid (E1/E2 of the paper)
    grid = [
        # (view, budget, gamma, force)
        ("global", 0.20, 0.0, False),   # M^G baseline
        ("global", 0.20, 0.0, True),    # FuelEU forcing at 20%
        ("global", 0.20, 0.5, True),
        ("global", 0.20, 1.0, True),
        ("global", 0.10, 0.0, False),   # M^G baseline 10%
        ("global", 0.10, 0.0, True),    # FuelEU forcing at 10%
        ("eu", 0.20, 0.0, False),       # M^EU no-force
        ("eu", 0.20, 0.5, True),        # M^EU with recycle
    ]
    for view, b, g, force in grid:
        run_fueleu_constrained_milp(vessel_type, budget_fraction=b, gamma=g,
                                    view=view, force_fueleu=force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vessel-type", default=VESSEL_TYPE)
    parser.add_argument("--budget-fraction", type=float, default=0.20)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--ets-price", type=float, default=80.0)
    parser.add_argument("--view", default="global", choices=["global", "eu"])
    parser.add_argument("--no-force-fueleu", action="store_true")
    args = parser.parse_args()
    if args.view != "global" or args.gamma != 0.0 or args.no_force_fueleu:
        run_fueleu_constrained_milp(args.vessel_type, budget_fraction=args.budget_fraction,
                                    gamma=args.gamma, ets_price=args.ets_price,
                                    view=args.view, force_fueleu=not args.no_force_fueleu)
    else:
        main(args.vessel_type)
