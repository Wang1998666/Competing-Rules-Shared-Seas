"""
Module 06 (v4): Network Diagnostic Indices
===========================================
Computes the five original indices from §4.7 of the paper:
  - DAI: Decarbonization Accessibility Index (per corridor)
  - DDR: Decarbonization Desert Rate (network-level)
  - CCI: Corridor Complementarity Index (pairwise, Jaccard)
  - PGI: Policy Gap Index (per corridor)
  - SRI: Siting Robustness Index (per port)

v4: paths point to processed_v2 / milp_results_v2 / diagnostics_v2; DAI/PGI
columns are written back to corridors_container.parquet so fig5 panels (a)(b)
render. Usage:
    python 06_diagnostics.py [--vessel-type container|tanker]
"""

import sys
import logging
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    OUTPUT_DIR, PROCESSED_DIR, RESULTS_DIR, EMISSION_PARAMS, MILP_PARAMS,
    VESSEL_TYPE, get_data_paths,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DIAG_DIR = OUTPUT_DIR / "diagnostics_v5"
DIAG_DIR.mkdir(parents=True, exist_ok=True)


def haversine_nm(lon1, lat1, lon2, lat2):
    """Great-circle distance in nautical miles (vectorized)."""
    R_nm = 3440.065
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R_nm * np.arcsin(np.sqrt(a))


# ============================================================
# DAI: Decarbonization Accessibility Index
# ============================================================

def compute_dai(corridors: pd.DataFrame, ports: pd.DataFrame,
                M_ell_df: pd.DataFrame, fuel: str = "green_ammonia") -> pd.Series:
    """
    DAI_e = (1/|L_e|) * Σ_{l∈L_e} min(1, R_f / D_l) * |P_feas ∩ σ_e| / |σ_e|
    
    Simplified operationalization:
      DAI_e = range_feasibility × port_coverage
      
    where:
      range_feasibility = min(1, R_f / mean_distance_e)
      port_coverage = |feasible bunkering ports along route| / |all candidate ports along route|
    """
    R_f = EMISSION_PARAMS["fuel_range_nm"].get(fuel, 5000)

    # Range feasibility component
    range_feas = np.minimum(1.0, R_f / corridors["mean_distance"].clip(lower=1))

    # Port coverage component:
    # For each corridor, what fraction of candidate intermediate ports are
    # actually selected (upgraded)?
    selected_ports = set()
    sel_port_file = RESULTS_DIR / f"selected_ports_{corridors.attrs.get('vessel_type', 'container')}.csv"
    if sel_port_file.exists():
        sel_df = pd.read_csv(sel_port_file)
        selected_ports = set(sel_df["port_code"].tolist())

    # Count intermediate ports per corridor that are selected
    if len(M_ell_df) > 0 and len(selected_ports) > 0:
        M_ell_df = M_ell_df.copy()
        M_ell_df["is_selected"] = M_ell_df["intermediate_port"].isin(selected_ports)
        port_cov = M_ell_df.groupby("corridor_id")["is_selected"].mean()
        port_cov = corridors["corridor_id"].map(port_cov).fillna(0)
    else:
        # No intermediates needed (direct routes) → full coverage if endpoints selected
        port_cov = pd.Series(1.0, index=corridors.index)
        if len(selected_ports) > 0:
            endpoints_selected = (
                corridors["origin_port"].isin(selected_ports) &
                corridors["dest_port"].isin(selected_ports)
            )
            port_cov = endpoints_selected.astype(float)

    dai = range_feas * port_cov
    return dai


# ============================================================
# DDR: Decarbonization Desert Rate
# ============================================================

def compute_ddr(dai: pd.Series, threshold: float = 0.3) -> float:
    """
    DDR = |{e : DAI_e < threshold}| / |E|
    Fraction of corridors that are "decarbonization deserts".
    """
    return (dai < threshold).mean()


# ============================================================
# CCI: Corridor Complementarity Index
# ============================================================

def compute_cci(corridors: pd.DataFrame, M_ell_df: pd.DataFrame,
                top_n: int = 50) -> pd.DataFrame:
    """
    CCI_{e1,e2} = |σ_{e1} ∩ σ_{e2}| / |σ_{e1} ∪ σ_{e2}|  (Jaccard)
    
    where σ_e = set of ports serving corridor e (endpoints + intermediates).
    Computed for top-N corridors by emission.
    """
    top_corridors = corridors.nlargest(top_n, "E_e")
    top_ids = top_corridors["corridor_id"].tolist()

    # Build port set per corridor
    port_sets = {}
    for _, row in top_corridors.iterrows():
        cid = row["corridor_id"]
        s = {row["origin_port"], row["dest_port"]}
        if len(M_ell_df) > 0:
            inters = M_ell_df.loc[
                M_ell_df["corridor_id"] == cid, "intermediate_port"
            ].tolist()
            s.update(inters)
        port_sets[cid] = s

    # Pairwise Jaccard
    records = []
    for i, cid1 in enumerate(top_ids):
        for cid2 in top_ids[i + 1:]:
            s1, s2 = port_sets[cid1], port_sets[cid2]
            intersection = len(s1 & s2)
            union = len(s1 | s2)
            jaccard = intersection / union if union > 0 else 0.0
            records.append({
                "corridor_1": cid1,
                "corridor_2": cid2,
                "CCI": jaccard,
                "shared_ports": intersection,
            })

    cci_df = pd.DataFrame(records)
    return cci_df


# ============================================================
# PGI: Policy Gap Index
# ============================================================

def compute_pgi(corridors: pd.DataFrame, ports: pd.DataFrame) -> pd.Series:
    """
    PGI_e = 1 - (policy_coverage_e / optimal_coverage_e)
    
    Operationalization:
      optimal_coverage = 1 (corridor is activated in MILP solution)
      policy_coverage = fraction of corridor's ports that are in
                        jurisdictions with active carbon pricing (EU ETS, etc.)
    
    Simplified: use port country → EU/EEA flag as proxy for policy coverage.
    """
    # EU/EEA country codes (approximate)
    eu_countries = {
        "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
        "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
        "PL", "PT", "RO", "SK", "SI", "ES", "SE",
        "IS", "LI", "NO",  # EEA
    }

    # Map port → country
    port_country = ports.drop_duplicates("port_code").set_index("port_code")
    if "country_code" in port_country.columns:
        country_map = port_country["country_code"].to_dict()
    elif "ctry_code" in port_country.columns:
        country_map = port_country["ctry_code"].to_dict()
    else:
        # No country info → PGI = 0 (no policy gap measurable)
        logger.warning("  No country code in ports → PGI set to 0")
        return pd.Series(0.0, index=corridors.index)

    # Policy coverage: fraction of endpoints in EU/EEA
    origin_eu = corridors["origin_port"].map(
        lambda p: country_map.get(p, "") in eu_countries
    ).astype(float)
    dest_eu = corridors["dest_port"].map(
        lambda p: country_map.get(p, "") in eu_countries
    ).astype(float)

    policy_coverage = (origin_eu + dest_eu) / 2.0

    # Optimal coverage = 1 for activated corridors, else 0
    activated_file = RESULTS_DIR / "activated_corridors_container.csv"
    if activated_file.exists():
        activated = pd.read_csv(activated_file)
        activated_ids = set(activated["corridor_id"].tolist())
        optimal = corridors["corridor_id"].isin(activated_ids).astype(float)
    else:
        optimal = pd.Series(1.0, index=corridors.index)

    # PGI: gap between policy and optimal
    pgi = np.where(optimal > 0, 1.0 - policy_coverage, 0.0)
    return pd.Series(pgi, index=corridors.index)


# ============================================================
# SRI: Siting Robustness Index
# ============================================================

def compute_sri(ports: pd.DataFrame, corridors: pd.DataFrame,
                M_ell_df: pd.DataFrame) -> pd.Series:
    """
    SRI_p = 1 - (ΔObj_p / Obj_total)
    
    where ΔObj_p = objective loss when port p is removed from candidate set.
    High SRI → port is critical (removal causes large loss).
    
    Simplified proxy: SRI_p ∝ number of activated corridors that depend on p
    (as endpoint or intermediate), weighted by their W_e.
    """
    activated_file = RESULTS_DIR / "activated_corridors_container.csv"
    if not activated_file.exists():
        logger.warning("  No activated corridors file → SRI set to 0")
        return pd.Series(0.0, index=ports.index)

    activated = pd.read_csv(activated_file)
    activated_ids = set(activated["corridor_id"].tolist())
    act_corr = corridors[corridors["corridor_id"].isin(activated_ids)].copy()

    if len(act_corr) == 0:
        return pd.Series(0.0, index=ports.index)

    total_obj = act_corr["W_e"].sum() if "W_e" in act_corr.columns else 1.0

    # Count weighted dependence per port
    port_dependency = {}

    # As endpoint
    for _, row in act_corr.iterrows():
        w = row.get("W_e", 1.0)
        for p in [row["origin_port"], row["dest_port"]]:
            port_dependency[p] = port_dependency.get(p, 0) + w

    # As intermediate (from M_ell)
    if len(M_ell_df) > 0:
        inter_act = M_ell_df[M_ell_df["corridor_id"].isin(activated_ids)]
        corr_w = act_corr.set_index("corridor_id")["W_e"].to_dict()
        for _, row in inter_act.iterrows():
            p = row["intermediate_port"]
            w = corr_w.get(row["corridor_id"], 0)
            port_dependency[p] = port_dependency.get(p, 0) + w * 0.5  # half weight

    # Normalize: SRI_p = dependency_p / total_obj
    sri_values = ports["port_code"].map(
        lambda p: port_dependency.get(p, 0) / total_obj if total_obj > 0 else 0
    )
    return sri_values


# ============================================================
# Main
# ============================================================

def main(vessel_type: str):
    paths = get_data_paths(vessel_type)
    logger.info(f"Computing diagnostic indices: {paths['label_cn']}")

    # Load data
    corridors = pd.read_parquet(PROCESSED_DIR / f"corridors_{vessel_type}.parquet")
    ports = pd.read_parquet(PROCESSED_DIR / f"candidate_ports_{vessel_type}.parquet")
    M_ell_df = pd.read_parquet(PROCESSED_DIR / f"M_ell_{vessel_type}.parquet")
    corridors.attrs["vessel_type"] = vessel_type
    logger.info(f"  Loaded: {len(corridors)} corridors, {len(ports)} ports")

    # ---- DAI ----
    logger.info("\n[DAI] Decarbonization Accessibility Index")
    for fuel in ["green_ammonia", "green_methanol"]:
        dai = compute_dai(corridors, ports, M_ell_df, fuel)
        corridors[f"DAI_{fuel}"] = dai
        logger.info(f"  {fuel}: mean={dai.mean():.3f}, median={dai.median():.3f}, "
                    f"min={dai.min():.3f}, max={dai.max():.3f}")

    # ---- DDR ----
    logger.info("\n[DDR] Decarbonization Desert Rate")
    for threshold in [0.2, 0.3, 0.5]:
        ddr = compute_ddr(corridors["DAI_green_ammonia"], threshold)
        logger.info(f"  DDR (threshold={threshold}): {ddr:.2%}")
    corridors["is_desert"] = corridors["DAI_green_ammonia"] < 0.3

    # ---- CCI ----
    logger.info("\n[CCI] Corridor Complementarity Index (top-50)")
    cci_df = compute_cci(corridors, M_ell_df, top_n=50)
    if len(cci_df) > 0:
        logger.info(f"  Pairs computed: {len(cci_df):,}")
        logger.info(f"  Mean CCI: {cci_df['CCI'].mean():.4f}")
        logger.info(f"  Max CCI: {cci_df['CCI'].max():.4f}")
        logger.info(f"  Pairs with CCI > 0.1: {(cci_df['CCI'] > 0.1).sum()}")
        cci_df.to_csv(DIAG_DIR / f"cci_top50_{vessel_type}.csv", index=False)

    # ---- PGI ----
    logger.info("\n[PGI] Policy Gap Index")
    pgi = compute_pgi(corridors, ports)
    corridors["PGI"] = pgi
    logger.info(f"  Mean PGI: {pgi.mean():.3f}")
    logger.info(f"  Corridors with PGI > 0.5: {(pgi > 0.5).sum()}")

    # ---- SRI ----
    logger.info("\n[SRI] Siting Robustness Index")
    sri = compute_sri(ports, corridors, M_ell_df)
    ports["SRI"] = sri
    logger.info(f"  Mean SRI: {sri.mean():.4f}")
    logger.info(f"  Top-10 critical ports (highest SRI):")
    top_sri = ports.nlargest(10, "SRI")
    for _, row in top_sri.iterrows():
        logger.info(f"    {row['port_code']}: SRI={row['SRI']:.4f}")

    # ---- Save ----
    logger.info("\n[Saving]")
    corridors.to_parquet(PROCESSED_DIR / f"corridors_{vessel_type}.parquet", index=False)
    ports.to_parquet(PROCESSED_DIR / f"candidate_ports_{vessel_type}.parquet", index=False)

    # Summary CSV
    summary = {
        "vessel_type": vessel_type,
        "n_corridors": len(corridors),
        "DAI_mean_ammonia": corridors["DAI_green_ammonia"].mean(),
        "DAI_mean_methanol": corridors["DAI_green_methanol"].mean(),
        "DDR_0.3": compute_ddr(corridors["DAI_green_ammonia"], 0.3),
        "CCI_mean": cci_df["CCI"].mean() if len(cci_df) > 0 else 0,
        "PGI_mean": pgi.mean(),
        "SRI_mean": sri.mean(),
        "SRI_max": sri.max(),
    }
    pd.DataFrame([summary]).to_csv(DIAG_DIR / f"diagnostics_summary_{vessel_type}.csv", index=False)
    logger.info(f"  Saved: diagnostics_summary_{vessel_type}.csv")

    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSTICS SUMMARY")
    logger.info("=" * 60)
    for k, v in summary.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    return corridors, ports, cci_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vessel-type", default=VESSEL_TYPE)
    args = parser.parse_args()
    main(args.vessel_type)
