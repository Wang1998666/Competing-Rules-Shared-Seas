"""
Module 04 (v2): Corridor Construction & Bunkering Activation Logic
==================================================================
v2 changes vs v1 (Eq. 8–12, Eq. 17–18):

  * Undirected corridor merging: (o,d) ≡ (d,o); E_e, f_e, distances and
    adoption shares are aggregated over both directions (liner service runs
    both ways).
  * Emission-weighted incumbent fuel factor per corridor:
        EF̄_e = Σ_v E_odv·EF_{f(v)} / Σ_v E_odv
    and η_e = 1 − EF_alt / EF̄_e (Eq. 9).
  * Intermediate bunkering sets M_ℓ built from OBSERVED midway mooring ports
    (moorPortCode) and sequence-junction ports (vessel call chains), not from
    all geographically near ports (Eq. 10): M_ℓ ⊆ σ_e \ {p_k, p_{k+1}}.
  * Berth-based feasibility mask P^feas (Eq. 18): candidate ports are feasible
    if a container-berth file exists in D:\Data\船视宝地理信息数据\港口泊位,
    with a call-frequency fallback for missing berth files.
  * Physical port cost model (Eq. 17):
        c_p = CRF(r_d, T_h)·(C^base + C^depth_p + C^remote_p),   [M USD/yr]
    C^depth from container-berth count, C^remote from distance to the nearest
    of five global fuel-supply hubs.

Outputs (02_数据_output/processed_v2/):
  corridors_{vt}.parquet, candidate_ports_{vt}.parquet, M_ell_{vt}.parquet
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PROCESSED_DIR, BERTH_DIR, EMISSION_PARAMS, CORRIDOR_PARAMS, COST_PARAMS,
    VESSEL_TYPE, get_data_paths,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

R_EARTH_NM = 3440.065


def haversine_nm(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R_EARTH_NM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def crf(r_d: float, t_h: int) -> float:
    """Capital recovery factor (Eq. 17)."""
    return r_d * (1 + r_d) ** t_h / ((1 + r_d) ** t_h - 1)


def build_sequence_junctions(legs: pd.DataFrame, max_window: int = 4):
    """Build vessel call chains to identify sequence-junction ports.
    For each vessel's sorted port chain, windows of length w=2..max_window
    record (origin, dest, mid-port-tuple); the mid ports of window (i, i+w)
    are the observed intermediate calls between ports[i] and ports[i+w]
    (Eq. 10: M_ℓ ⊆ σ_e \ {p_k, p_{k+1}}). Returns
    DataFrame(origin_port, dest_port, mid_ports, n_occur)."""
    recs = []
    legs_sorted = legs.sort_values(["mmsi", "leg_start_time"])
    g = legs_sorted.groupby("mmsi")
    n_chain = 0
    for mmsi, sub in g:
        ports = sub["origin_port"].tolist() + [sub["dest_port"].iloc[-1]]
        if len(ports) < 3:
            continue
        n_chain += 1
        for w in range(2, min(max_window, len(ports)) + 1):
            for i in range(0, len(ports) - w):
                recs.append((ports[i], ports[i + w],
                             tuple(ports[i + 1:i + w])))
    if not recs:
        return pd.DataFrame(columns=["origin_port", "dest_port", "mid_ports"])
    df = pd.DataFrame(recs, columns=["origin_port", "dest_port", "mid_ports"])
    logger.info(f"  Call chains built for {n_chain:,} vessels; "
                f"window records: {len(df):,}")
    return df.groupby(["origin_port", "dest_port", "mid_ports"]).size() \
             .reset_index(name="n_occur")


def load_berth_feasibility(port_codes, call_counts):
    """Berth-based feasibility mask (Eq. 18).
    A port is feasible if it has a container berth file (berthType == 100);
    ports without a berth file remain feasible if they have >= threshold
    annual calls (data-coverage protection)."""
    container_type = CORRIDOR_PARAMS["container_berth_type"]
    threshold = CORRIDOR_PARAMS["berth_missing_call_threshold"]
    feas = {}
    n_berth_ok = n_fallback = 0
    for pc in port_codes:
        f = BERTH_DIR / f"{pc}.csv"
        ok = False
        n_berth = 0
        if f.exists():
            try:
                b = pd.read_csv(f, usecols=["berthType"], low_memory=False)
                n_berth = int((b["berthType"] == container_type).sum())
                ok = n_berth > 0
            except Exception:
                ok = False
        if ok:
            n_berth_ok += 1
        else:
            # fallback: busy ports kept even if berth file missing/unreadable
            if call_counts.get(pc, 0) >= threshold:
                ok = True
                n_fallback += 1
        feas[pc] = (ok, n_berth)
    logger.info(f"  Feasibility: {n_berth_ok} ports with container berths, "
                f"{n_fallback} kept by call-frequency fallback")
    return feas


def assign_port_costs(port_df, feas):
    """Physical cost model (Eq. 17): c_p = CRF·(C_base + C_depth + C_remote).
    Units: M USD/yr. Also stores capital components for transparency."""
    r_d = COST_PARAMS["discount_rate"]
    t_h = COST_PARAMS["planning_horizon_years"]
    crf_val = crf(r_d, t_h)
    logger.info(f"  CRF({r_d}, {t_h}y) = {crf_val:.4f}")

    hub_codes = [c for c in COST_PARAMS["fuel_hub_ports"] if c in set(port_df["port_code"])]
    hub_lon = port_df.set_index("port_code").loc[hub_codes, "port_lon"].values
    hub_lat = port_df.set_index("port_code").loc[hub_codes, "port_lat"].values

    lon = port_df["port_lon"].values
    lat = port_df["port_lat"].values
    n_berths = port_df["n_container_berths"].values

    # Remoteness: distance to nearest fuel-supply hub (nm)
    d_min = np.full(len(port_df), np.inf)
    for hlon, hlat in zip(hub_lon, hub_lat):
        d = haversine_nm(lon, lat, np.full(len(lon), hlon), np.full(len(lat), hlat))
        d_min = np.minimum(d_min, d)
    C_remote = np.minimum(COST_PARAMS["C_remote_rate"] * d_min,
                          COST_PARAMS["C_remote_cap"])

    # Depth/civil premium from container-berth count (proxy for marginal depth)
    C_depth = np.clip(COST_PARAMS["C_depth_penalty"] - 5.0 * n_berths, 0,
                      COST_PARAMS["C_depth_penalty"])

    C_base = np.full(len(port_df), COST_PARAMS["C_base"])
    C_cap = C_base + C_depth + C_remote
    cost_p = crf_val * C_cap

    port_df["C_base_MUSD"] = C_base
    port_df["C_depth_MUSD"] = C_depth
    port_df["C_remote_MUSD"] = C_remote
    port_df["C_cap_MUSD"] = C_cap
    port_df["cost_p"] = cost_p  # M USD/yr (annualized)
    return port_df


def build_corridors(vessel_type: str):
    paths = get_data_paths(vessel_type)
    is_liner = vessel_type in CORRIDOR_PARAMS["liner_vessel_types"]
    logger.info(f"Corridor construction v2: {paths['label_cn']} "
                f"({'liner' if is_liner else 'tramp'})")

    emissions = pd.read_parquet(PROCESSED_DIR / f"emissions_{vessel_type}.parquet")
    legs = pd.read_parquet(PROCESSED_DIR / f"legs_{vessel_type}.parquet")
    ports = pd.read_parquet(PROCESSED_DIR / "ports_global.parquet")
    logger.info(f"  Loaded {len(emissions):,} legs, {len(ports):,} ports")

    ports_dedup = ports.drop_duplicates(subset="port_code", keep="first")
    port_coords = ports_dedup.set_index("port_code")[["port_lon", "port_lat"]].to_dict("index")

    # ================================================================
    # Step 1: Undirected corridor aggregation (v2)
    # ================================================================
    logger.info("\n[Step 1] Undirected corridor extraction")
    n_min = CORRIDOR_PARAMS["min_annual_frequency"]
    em = emissions.copy()
    # drop legs with missing port codes (cannot be corridor-attributed)
    em = em.dropna(subset=["origin_port", "dest_port"])
    em["origin_port"] = em["origin_port"].astype(str)
    em["dest_port"] = em["dest_port"].astype(str)
    em["p1"] = np.minimum(em["origin_port"], em["dest_port"])
    em["p2"] = np.maximum(em["origin_port"], em["dest_port"])

    # Voyage weight (each leg = one voyage in the direction travelled)
    em["w"] = 1.0
    agg = em.groupby(["p1", "p2"]).agg(
        frequency=("w", "sum"),
        n_vessels=("mmsi", "nunique"),
        total_emission=("emission_tco2e", "sum"),
        # emission-weighted distance, speed, EF (v2: Eq. 9 weights)
        mean_distance=("sail_distance_nm", lambda s: np.average(s, weights=em.loc[s.index, "emission_tco2e"])),
        mean_speed=("avg_speed_kn", lambda s: np.average(s, weights=em.loc[s.index, "emission_tco2e"])),
        mean_load_factor=("load_factor", lambda s: np.average(s, weights=em.loc[s.index, "emission_tco2e"])),
        ef_weighted=("ef_wtw", lambda s: np.average(s, weights=em.loc[s.index, "emission_tco2e"])),
        n_legs_direct=("is_direct", lambda s: int((em.loc[s.index, "is_direct"] == 1).sum())),
    ).reset_index()

    agg = agg.rename(columns={"p1": "origin_port", "p2": "dest_port",
                              "ef_weighted": "EF_avg_e"})
    corridors = agg[agg["frequency"] >= n_min].copy()
    corridors = corridors.sort_values("total_emission", ascending=False).reset_index(drop=True)
    corridors["corridor_id"] = range(len(corridors))

    logger.info(f"  Total undirected OD pairs: {len(agg):,}")
    logger.info(f"  Corridors (freq >= {n_min}): {len(corridors):,}")
    cov = corridors["total_emission"].sum() / em["emission_tco2e"].sum() * 100
    logger.info(f"  Emission coverage: {cov:.1f}%")

    corridors["E_e"] = corridors["total_emission"]

    # ================================================================
    # Step 2: Emission-weighted fuel reduction rate (Eq. 9, v2)
    # ================================================================
    logger.info("\n[Step 2] Corridor attributes")
    ef_current = corridors["EF_avg_e"].values
    for fuel, ef_alt in EMISSION_PARAMS["ef_wtw"].items():
        if fuel in ["green_methanol", "green_ammonia", "LNG_HP"]:
            fuel_label = fuel.replace("_HP", "")
            corridors[f"eta_{fuel_label}"] = 1.0 - ef_alt / ef_current

    # Adoption share alpha_e (fleet age <= 15, voyage-weighted)
    vessels = pd.read_parquet(PROCESSED_DIR / f"vessels_{vessel_type}.parquet")
    if "build_year" in vessels.columns:
        vessel_age = vessels[["mmsi", "build_year"]].copy()
        vessel_age["retrofit_eligible"] = (2024 - vessel_age["build_year"]) <= 15
        legs_age = em[["mmsi", "p1", "p2", "w"]].merge(
            vessel_age[["mmsi", "retrofit_eligible"]], on="mmsi", how="left")
        alpha = legs_age.groupby(["p1", "p2"]).apply(
            lambda s: np.average(s["retrofit_eligible"].fillna(False).astype(float),
                                 weights=s["w"]), include_groups=False
        ).reset_index(name="alpha_e")
        alpha = alpha.rename(columns={"p1": "origin_port", "p2": "dest_port"})
        corridors = corridors.merge(alpha, on=["origin_port", "dest_port"], how="left")
        corridors["alpha_e"] = corridors["alpha_e"].fillna(0.5)
    else:
        corridors["alpha_e"] = 0.5

    # Abatement weight for reference fuel (methanol for liners)
    default_fuel = "green_methanol" if is_liner else "green_ammonia"
    eta_col = f"eta_{default_fuel}"
    corridors["W_e"] = corridors["E_e"] * corridors[eta_col] * corridors["alpha_e"]
    logger.info(f"  Mean alpha_e: {corridors['alpha_e'].mean():.3f}")
    logger.info(f"  Mean EF_avg_e: {corridors['EF_avg_e'].mean():.1f} g/kWh")
    logger.info(f"  Default fuel: {default_fuel}; total W_e: "
                f"{corridors['W_e'].sum()/1e6:.2f} Mt")

    # ================================================================
    # Step 3: Coordinates & range logic
    # ================================================================
    logger.info("\n[Step 3] Bunkering activation logic")
    fuel_ranges = EMISSION_PARAMS["fuel_range_nm"]

    corridors["origin_lon"] = corridors["origin_port"].map(
        lambda p: port_coords.get(p, {}).get("port_lon", np.nan))
    corridors["origin_lat"] = corridors["origin_port"].map(
        lambda p: port_coords.get(p, {}).get("port_lat", np.nan))
    corridors["dest_lon"] = corridors["dest_port"].map(
        lambda p: port_coords.get(p, {}).get("port_lon", np.nan))
    corridors["dest_lat"] = corridors["dest_port"].map(
        lambda p: port_coords.get(p, {}).get("port_lat", np.nan))
    corridors["gc_distance_nm"] = haversine_nm(
        corridors["origin_lon"].values, corridors["origin_lat"].values,
        corridors["dest_lon"].values, corridors["dest_lat"].values)

    for fuel, R_f in fuel_ranges.items():
        if R_f == float("inf"):
            corridors[f"range_violating_{fuel}"] = False
            corridors[f"k_e_{fuel}"] = 0
        else:
            viol = corridors["mean_distance"] > R_f
            corridors[f"range_violating_{fuel}"] = viol
            corridors[f"k_e_{fuel}"] = (np.ceil(corridors["mean_distance"] / R_f)
                                        .astype(int) - 1).clip(lower=0)
    logger.info(f"  Range-violating (ammonia 5000nm): "
                f"{corridors['range_violating_green_ammonia'].sum():,}")
    logger.info(f"  Range-violating (methanol 8000nm): "
                f"{corridors['range_violating_green_methanol'].sum():,}")

    # ================================================================
    # Step 4: Sequence-based intermediate bunkering sets (Eq. 10, v2)
    # ================================================================
    logger.info("\n[Step 4] Sequence-based intermediate ports M_ell")
    M_ell_records = []

    # (a) midway mooring ports from leg records
    moor = legs[["mmsi", "origin_port", "dest_port", "moor_port_code"]].copy()
    moor = moor.dropna(subset=["moor_port_code"])
    moor = moor[moor["moor_port_code"] != ""]
    if len(moor):
        moor["p1"] = np.minimum(moor["origin_port"], moor["dest_port"])
        moor["p2"] = np.maximum(moor["origin_port"], moor["dest_port"])
        moor_count = (moor.groupby(["p1", "p2", "moor_port_code"])
                      .size().reset_index(name="n_moor"))

    # (b) junction ports from vessel call chains (multi-window)
    junction = build_sequence_junctions(legs, max_window=4)
    # expand mid-port tuples into (o, d, mid) rows
    if len(junction):
        jrecs = []
        for _, r in junction.iterrows():
            for mid in r["mid_ports"]:
                jrecs.append((r["origin_port"], r["dest_port"], mid, r["n_occur"]))
        junction = pd.DataFrame(jrecs, columns=["origin_port", "dest_port", "junction_port", "n_occur"])
        # normalize direction (corridors are undirected)
        junction["p1"] = np.minimum(junction["origin_port"], junction["dest_port"])
        junction["p2"] = np.maximum(junction["origin_port"], junction["dest_port"])
    logger.info(f"  Sequence-junction records: {len(junction):,}")

    for _, row in corridors.iterrows():
        cid = row["corridor_id"]
        o, d = row["origin_port"], row["dest_port"]
        if pd.isna(row["origin_lon"]) or pd.isna(row["dest_lon"]):
            continue

        # intermediate candidates: mooring ports + junction ports
        cand = set()
        if len(moor):
            sub = moor_count[(moor_count["p1"] == o) & (moor_count["p2"] == d)]
            cand |= set(sub["moor_port_code"].tolist())
        sub_j = junction[(junction["p1"] == o) & (junction["p2"] == d)]
        cand |= set(sub_j["junction_port"].tolist())
        cand = {p for p in cand if p not in (o, d) and p in port_coords}

        for fuel, R_f in fuel_ranges.items():
            if R_f == float("inf") or not row[f"range_violating_{fuel}"]:
                continue
            # ports within range of BOTH endpoints (Eq. 10: split long leg)
            for p in cand:
                p_lon, p_lat = port_coords[p]["port_lon"], port_coords[p]["port_lat"]
                d1 = haversine_nm(row["origin_lon"], row["origin_lat"], p_lon, p_lat)
                d2 = haversine_nm(p_lon, p_lat, row["dest_lon"], row["dest_lat"])
                if d1 <= R_f and d2 <= R_f:
                    M_ell_records.append({
                        "corridor_id": cid, "origin_port": o, "dest_port": d,
                        "intermediate_port": p, "fuel": fuel,
                        "d_origin_nm": d1, "d_dest_nm": d2})

    M_ell_df = pd.DataFrame(M_ell_records) if M_ell_records else pd.DataFrame(
        columns=["corridor_id", "origin_port", "dest_port", "intermediate_port",
                 "fuel", "d_origin_nm", "d_dest_nm"])
    if len(M_ell_df):
        logger.info(f"  M_ell entries: {len(M_ell_df):,}; corridors covered: "
                    f"{M_ell_df['corridor_id'].nunique():,}")
        for fuel in ["green_ammonia", "green_methanol"]:
            n = M_ell_df.loc[M_ell_df["fuel"] == fuel, "corridor_id"].nunique()
            logger.info(f"    {fuel}: {n:,} corridors with intermediates")
    else:
        logger.info("  No intermediate bunkering candidates found.")

    # ================================================================
    # Step 5: Candidate ports, feasibility mask & costs (Eq. 17–18)
    # ================================================================
    logger.info("\n[Step 5] Candidate ports, feasibility & cost")
    candidate_ports = set(corridors["origin_port"]) | set(corridors["dest_port"])
    if len(M_ell_df):
        candidate_ports |= set(M_ell_df["intermediate_port"].unique())

    port_df = ports[ports["port_code"].isin(candidate_ports)].copy()
    port_df = port_df.drop_duplicates("port_code", keep="first")

    # endpoint counts
    endpoint_counts = pd.concat([
        corridors[["origin_port"]].rename(columns={"origin_port": "port_code"}),
        corridors[["dest_port"]].rename(columns={"dest_port": "port_code"}),
    ]).groupby("port_code").size().reset_index(name="n_corridors_endpoint")
    port_df = port_df.merge(endpoint_counts, on="port_code", how="left")
    port_df["n_corridors_endpoint"] = port_df["n_corridors_endpoint"].fillna(0).astype(int)

    # call counts from leg records (for fallback)
    call_counts = pd.concat([
        em[["origin_port"]].rename(columns={"origin_port": "port_code"}),
        em[["dest_port"]].rename(columns={"dest_port": "port_code"}),
    ]).groupby("port_code").size().to_dict()

    # feasibility mask (Eq. 18)
    feas = load_berth_feasibility(list(port_df["port_code"]), call_counts)
    port_df["n_container_berths"] = port_df["port_code"].map(lambda p: feas[p][1])
    port_df["is_feasible"] = port_df["port_code"].map(lambda p: feas[p][0])
    logger.info(f"  Candidate ports: {len(port_df):,}; feasible: "
                f"{port_df['is_feasible'].sum():,} "
                f"({port_df['is_feasible'].mean()*100:.1f}%)")

    port_df["is_intermediate"] = port_df["port_code"].isin(
        set(M_ell_df["intermediate_port"]) if len(M_ell_df) else set())

    # physical costs (Eq. 17)
    port_df = assign_port_costs(port_df, feas)
    logger.info(f"  Cost pool Σc_p = {port_df['cost_p'].sum():,.1f} M USD/yr; "
                f"mean c_p = {port_df['cost_p'].mean():.2f} M USD/yr")

    # ================================================================
    # Save
    # ================================================================
    corridors.to_parquet(PROCESSED_DIR / f"corridors_{vessel_type}.parquet", index=False)
    port_df.to_parquet(PROCESSED_DIR / f"candidate_ports_{vessel_type}.parquet", index=False)
    M_ell_df.to_parquet(PROCESSED_DIR / f"M_ell_{vessel_type}.parquet", index=False)

    logger.info(f"\n  corridors_{vessel_type}.parquet: {len(corridors)} corridors")
    logger.info(f"  candidate_ports_{vessel_type}.parquet: {len(port_df)} ports")
    logger.info(f"  M_ell_{vessel_type}.parquet: {len(M_ell_df)} entries")

    logger.info("\n" + "=" * 60)
    logger.info("CORRIDOR SUMMARY (v2)")
    logger.info("=" * 60)
    logger.info(f"  Corridors: {len(corridors):,}")
    logger.info(f"  Total emission: {corridors['E_e'].sum()/1e6:.2f} Mt CO2e")
    logger.info(f"  Abatement potential (W_e, methanol): {corridors['W_e'].sum()/1e6:.2f} Mt")
    logger.info("=" * 60)
    return corridors, port_df, M_ell_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vessel-type", default=VESSEL_TYPE)
    args = parser.parse_args()
    build_corridors(args.vessel_type)
