"""
Module 02 (v5): Voyage-Leg Emission Estimation — Revised Accounting
===================================================================
IMO bottom-up approach with the four corrections of the emission-gap
diagnosis (emission_gap_revision/emission_gap_revision.html):

  C1  EF_WtW in g CO2e/kWh = EF_MJ x 3.6/eta (eta = 0.50 uniform, main
      caliber) — values corrected in config.py (v2-v4 carried the
      g/MJ x 3.6 values that implicitly assumed 100% engine efficiency).
  C2  Auxiliary-engine & boiler uplift  (1 + aux_engine_fraction = 1.15).
  C3  Port maneuvering / anchorage uplift (1 + port_time_uplift = 1.05).
  C4  Trajectory-resolved load factor: per-leg underway time integral of
      clip[(SOG/v_design)^3, 0.2, 1] from module 02b replaces the
      leg-average cube law (Jensen gap). Legs without a matched
      trajectory integral fall back to the leg-average form.

    E_odv = P_ME(75%) x LF_eff x T_eff x EF_WtW x 1e-6
            x (1 + aux) x (1 + port)                     [tonnes CO2e]

Distances and the fallback sailing time follow v2 unchanged (provider
AIS distance preferred, regionalized detour x GC fallback).

Outputs: 02_数据_output/processed_v5/emissions_{vt}.parquet,
         emission_stats_{vt}.csv
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PROCESSED_DIR, EMISSION_PARAMS, VESSEL_TYPE, get_data_paths,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

R_EARTH_NM = 3440.065


def haversine_nm(lon1, lat1, lon2, lat2):
    """Great-circle distance in nautical miles (vectorized)."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R_EARTH_NM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def regional_detour_factor(lon_mid, lat_mid) -> np.ndarray:
    """Regionalized detour factor by leg-midpoint region (v2, Eq. 4 fallback)."""
    factor = np.full(np.asarray(lon_mid).shape, EMISSION_PARAMS["gc_detour_factor"],
                     dtype=float)
    for region in EMISSION_PARAMS["detour_regions"]:
        lon_ok = (lon_mid >= region["lon"][0]) & (lon_mid <= region["lon"][1])
        lat_ok = (lat_mid >= region["lat"][0]) & (lat_mid <= region["lat"][1])
        factor = np.where(lon_ok & lat_ok, region["factor"], factor)
    return factor


def infer_ballast_flag(legs: pd.DataFrame, vessel_type: str) -> np.ndarray:
    """Ballast flag; containerships: gamma = 1 (no ballast legs)."""
    n = len(legs)
    ballast = np.zeros(n, dtype=bool)
    if vessel_type == "tanker":
        if "start_draught_m" in legs.columns and "end_draught_m" in legs.columns:
            max_draught = np.maximum(legs["start_draught_m"].fillna(0).values,
                                     legs["end_draught_m"].fillna(0).values)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(max_draught > 0,
                                 legs["start_draught_m"].fillna(0).values / max_draught,
                                 1.0)
            ballast = ratio < 0.6
    return ballast


def assign_emission_factor(vessels: pd.DataFrame) -> pd.Series:
    """Per-vessel WtW emission factor (C1-corrected values from config);
    LNG/dual-fuel engines identified from engine model/designer strings."""
    ef = EMISSION_PARAMS["ef_wtw"]
    default_ef = ef["HFO"]
    ef_series = pd.Series(default_ef, index=vessels.index, name="ef_wtw")
    if "engine_model" in vessels.columns:
        lng_mask = vessels["engine_model"].fillna("").str.contains(
            "LNG|DF|dual.fuel|ME-GI|X-DF", case=False, na=False)
        ef_series[lng_mask] = ef["LNG_HP"]
        logger.info(f"  LNG/dual-fuel vessels (engine model): {lng_mask.sum()}")
    if "engine_designer" in vessels.columns:
        df_mask = vessels["engine_designer"].fillna("").str.contains(
            "WinGD|Wartsila.*DF", case=False, na=False)
        ef_series[df_mask & (ef_series == default_ef)] = ef["LNG_LP"]
    return ef_series


def estimate_emissions(vessel_type: str):
    """Run the v5 emission estimation pipeline (C1-C4)."""
    paths = get_data_paths(vessel_type)
    logger.info(f"Emission estimation v5: {paths['label_cn']}")

    vessels = pd.read_parquet(PROCESSED_DIR / f"vessels_{vessel_type}.parquet")
    legs = pd.read_parquet(PROCESSED_DIR / f"legs_{vessel_type}.parquet")
    logger.info(f"  Vessels: {len(vessels):,}, Legs: {len(legs):,}")

    merge_cols = ["mmsi", "p_me_75_kw", "design_speed_kn", "dwt", "teu",
                  "build_year", "engine_model", "engine_designer", "vessel_type"]
    merge_cols = [c for c in merge_cols if c in vessels.columns]
    legs = legs.merge(vessels[merge_cols], on="mmsi", how="left", suffixes=("", "_vessel"))

    # --- Filter: keep legs with required fields (v2 semantics) ---
    required = ["p_me_75_kw", "avg_speed_kn", "sail_distance_nm"]
    mask = legs[required].notna().all(axis=1)
    mask &= (legs["avg_speed_kn"] > 0) & (legs["avg_speed_kn"] <= 35)
    mask &= (legs["sail_distance_nm"] > 0) & (legs["sail_distance_nm"] <= 25000)
    if "design_speed_kn" in legs.columns:
        median_speed = legs.loc[legs["design_speed_kn"] > 0, "design_speed_kn"].median()
        legs["design_speed_kn"] = legs["design_speed_kn"].fillna(median_speed)
        legs.loc[legs["design_speed_kn"] <= 0, "design_speed_kn"] = median_speed
        mask &= (legs["design_speed_kn"] > 0)
    if "is_ais_abnormal" in legs.columns:
        n_abn = int((legs["is_ais_abnormal"] == 1).sum())
        if n_abn:
            mask &= (legs["is_ais_abnormal"] != 1)
            logger.info(f"  Dropped {n_abn:,} provider-flagged abnormal legs")

    n_dropped = (~mask).sum()
    legs = legs[mask].copy().reset_index(drop=True)
    logger.info(f"  Legs after filtering: {len(legs):,} (dropped {n_dropped:,})")

    # --- Distance: provider AIS distance preferred; GC × regional detour fallback
    if "origin_lon" in legs.columns and "dest_lon" in legs.columns:
        mid_lon = (legs["origin_lon"].values + legs["dest_lon"].values) / 2.0
        mid_lat = (legs["origin_lat"].values + legs["dest_lat"].values) / 2.0
        detour = regional_detour_factor(mid_lon, mid_lat)
        gc_nm = haversine_nm(legs["origin_lon"].values, legs["origin_lat"].values,
                             legs["dest_lon"].values, legs["dest_lat"].values)
        legs["detour_factor"] = detour
        legs["gc_distance_nm"] = gc_nm
        missing_dist = legs["sail_distance_nm"].isna() | (legs["sail_distance_nm"] <= 0)
        legs["sail_distance_nm"] = legs["sail_distance_nm"].where(~missing_dist,
                                                                  detour * gc_nm)
        n_filled = int(missing_dist.sum())
        if n_filled:
            logger.info(f"  GC×detour filled {n_filled:,} missing distances")
    else:
        legs["detour_factor"] = EMISSION_PARAMS["gc_detour_factor"]

    # --- Sailing time (fallback time base + output schema, v2 semantics) ---
    legs["sailing_time_h"] = legs["sail_distance_nm"] / legs["avg_speed_kn"]
    if "sail_duration_h" in legs.columns:
        computed = legs["sailing_time_h"]
        provided = legs["sail_duration_h"]
        use_provided = (provided > 0) & (
            np.abs(provided - computed) / computed.clip(lower=0.1) < 0.5)
        legs["sailing_time_h"] = np.where(use_provided, provided, computed)

    # --- Ballast correction (containerships: gamma = 1) ---
    gamma_base = EMISSION_PARAMS["ballast_correction"].get(vessel_type, 1.0)
    ballast_flags = infer_ballast_flag(legs, vessel_type)
    gamma_arr = np.where(ballast_flags, gamma_base, 1.0)
    legs["is_ballast"] = ballast_flags

    # --- C4: trajectory-resolved load-factor integrals ---
    use_traj = EMISSION_PARAMS.get("use_traj_load_factor", False)
    integ = None
    if use_traj:
        integ_path = PROCESSED_DIR / f"traj_leg_integrals_{vessel_type}.parquet"
        if integ_path.exists():
            integ = pd.read_parquet(integ_path)
            key = ["mmsi", "origin_port", "dest_port", "leg_start_time"]
            legs = legs.merge(integ[key + ["traj_integ_h", "traj_t_under_h"]],
                              on=key, how="left")
            logger.info(f"  C4: trajectory integrals merged "
                        f"({int(integ['traj_integ_h'].notna().sum()):,} legs with "
                        f"integral of {len(integ):,} in table)")
        else:
            logger.warning(f"  C4 requested but {integ_path} missing — "
                           f"run 02b_sea_speed_from_ais.py first; using fallback")

    if use_traj and "traj_integ_h" in legs.columns:
        has_traj = legs["traj_integ_h"].notna() & (legs["traj_integ_h"] > 0)
    else:
        has_traj = pd.Series(False, index=legs.index)
    if has_traj.any():
        legs["lf_source"] = np.where(has_traj, "traj", "legavg")
        n_traj = int(has_traj.sum())
        logger.info(f"  C4 coverage: {n_traj:,} legs ({n_traj/len(legs)*100:.1f}%) "
                    f"use trajectory integrals; {len(legs)-n_traj:,} fall back")
    else:
        legs["lf_source"] = "legavg"
        legs["traj_integ_h"] = np.nan
        legs["traj_t_under_h"] = np.nan

    # --- Leg-average load factor (fallback form + reporting) ---
    exponent = EMISSION_PARAMS["load_factor_exponent"]
    lf_min = EMISSION_PARAMS["load_factor_min"]
    lf_max = EMISSION_PARAMS["load_factor_max"]
    speed_ratio = np.where(legs["design_speed_kn"].values > 0,
                           legs["avg_speed_kn"].values / legs["design_speed_kn"].values,
                           0.0)
    lf_raw = gamma_arr * np.power(speed_ratio, exponent)
    lf_legavg = np.clip(lf_raw, lf_min, lf_max)

    # Effective time-integral hours: trajectory integral where available
    integ_h = legs["traj_integ_h"].values.astype(float)
    t_under = legs["traj_t_under_h"].values.astype(float)
    use_traj_arr = has_traj.values.astype(float)
    eff_hours = np.where(use_traj_arr > 0, integ_h, lf_legavg * legs["sailing_time_h"].values)
    # Mean-underway LF for reporting (traj legs: integral/underway time)
    lf_traj_mean = np.where(t_under > 0, integ_h / t_under, np.nan)
    legs["load_factor"] = np.where(use_traj_arr > 0, lf_traj_mean, lf_legavg)

    # --- Emission factor assignment (per vessel, C1 values) ---
    vessels_ef = vessels.copy()
    vessels_ef["ef_wtw"] = assign_emission_factor(vessels_ef)
    ef_map = vessels_ef.set_index("mmsi")["ef_wtw"].to_dict()
    legs["ef_wtw"] = legs["mmsi"].map(ef_map).fillna(EMISSION_PARAMS["ef_wtw"]["HFO"])

    # --- Emissions: E = P x LF_eff x T_eff x EF (C1) x (1+aux) (C2) x (1+port) (C3)
    aux_fraction = EMISSION_PARAMS.get("aux_engine_fraction", 0.0)
    port_uplift = EMISSION_PARAMS.get("port_time_uplift", 0.0)
    legs["emission_tco2e"] = (legs["p_me_75_kw"].values * eff_hours *
                              legs["ef_wtw"].values * 1e-6)
    legs["emission_tco2e"] *= (1.0 + aux_fraction) * (1.0 + port_uplift)
    logger.info(f"  C2 auxiliary uplift: +{aux_fraction*100:.0f}%  "
                f"C3 port uplift: +{port_uplift*100:.0f}%")

    # --- Quality checks ---
    total = legs["emission_tco2e"].sum()
    logger.info(f"\n  EMISSION RESULTS (v5):")
    logger.info(f"    Total legs with emissions: {len(legs):,}")
    logger.info(f"    Total emissions: {total:,.0f} t CO2e ({total/1e6:.1f} Mt)")
    logger.info(f"    Mean per leg: {legs['emission_tco2e'].mean():.2f} t CO2e")
    vessel_annual = legs.groupby("mmsi")["emission_tco2e"].sum()
    logger.info(f"    Vessels with emissions: {len(vessel_annual):,}")
    if use_traj_arr.any():
        e_traj = legs.loc[legs["lf_source"] == "traj", "emission_tco2e"].sum()
        logger.info(f"    Emission share from trajectory legs: "
                    f"{e_traj/total*100:.1f}%")

    # --- Save ---
    out_cols = [
        "mmsi", "origin_port", "dest_port", "origin_port_name", "dest_port_name",
        "origin_country", "dest_country",
        "leg_start_time", "leg_end_time",
        "sail_distance_nm", "avg_speed_kn", "sailing_time_h",
        "start_draught_m", "end_draught_m", "is_ballast",
        "is_canal", "is_direct", "moor_port_code",
        "origin_lon", "origin_lat", "dest_lon", "dest_lat",
        "p_me_75_kw", "design_speed_kn", "load_factor", "ef_wtw",
        "lf_source", "traj_integ_h", "traj_t_under_h",
        "emission_tco2e", "vessel_type",
    ]
    out_cols = [c for c in out_cols if c in legs.columns]
    legs_out = legs[out_cols].copy()
    out_path = PROCESSED_DIR / f"emissions_{vessel_type}.parquet"
    legs_out.to_parquet(out_path, index=False)
    logger.info(f"  Saved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    ew_lf = float((legs["load_factor"] * legs["emission_tco2e"]).sum() / total)
    stats = {
        "vessel_type": vessel_type,
        "version": "v5",
        "n_vessels": int(legs["mmsi"].nunique()),
        "n_legs": int(len(legs)),
        "n_od_pairs": int(legs.groupby(["origin_port", "dest_port"]).ngroups),
        "total_emission_tco2e": float(total),
        "mean_leg_emission": float(legs["emission_tco2e"].mean()),
        "mean_load_factor": float(legs["load_factor"].mean()),
        "emission_weighted_load_factor": ew_lf,
        "ballast_fraction": float(legs["is_ballast"].mean()),
        "aux_fraction": float(aux_fraction),
        "port_uplift": float(port_uplift),
        "ef_eta": float(EMISSION_PARAMS.get("ef_eta", 0.5)),
        "traj_leg_fraction": float((legs["lf_source"] == "traj").mean()),
        "traj_emission_fraction": float(e_traj / total) if use_traj_arr.any() else 0.0,
    }
    stats_df = pd.DataFrame([stats])
    stats_path = PROCESSED_DIR / f"emission_stats_{vessel_type}.csv"
    stats_df.to_csv(stats_path, index=False)
    logger.info(f"  Stats: {stats_path}")
    return legs_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emission estimation v5 (C1-C4)")
    parser.add_argument("--vessel-type", default=VESSEL_TYPE)
    args = parser.parse_args()
    estimate_emissions(args.vessel_type)
