r"""
GSC Planning Pipeline v5 — Emission Accounting Revision
========================================================
v5 implements the corrections identified in the emission-gap diagnosis
(emission_gap_revision/emission_gap_revision.html, 2026-08-23):

  C1  EF unit-conversion fix (ROOT CAUSE, x2.0): the v2-v4 ef_wtw values
      equalled EF_MJ x 3.6, which implicitly assumed 100% main-engine
      thermal efficiency. The power in E = P_ME x LF x T x EF is SHAFT
      power, so the factor must be EF_kWh = EF_MJ x SEC = EF_MJ x 3.6/eta.
      Main caliber uses a uniform eta = 0.50 (typical large two-stroke
      diesel SFC ~ 175 g/kWh, LHV 40.5 MJ/kg), i.e. EF_kWh = EF_MJ x 7.2.
      Per-fuel eta (0.47-0.51, IMO 4th GHG Study by engine type) is kept
      for sensitivity analysis only, so that all fuel-share ratios (eta_e,
      governance shares, MILP argmax) remain exactly invariant.

  C2  Auxiliary-engine & boiler uplift (x1.15): aux_engine_fraction = 0.15
      (IMO 4th GHG Study, containerships: ME ~ 80-85% of energy; the
      previous 0.0 with "slightly understates" wording was untenable).

  C3  Port-time uplift (x1.05): port maneuvering / anchorage main-engine
      activity outside the port-to-port leg table.

  C4  Trajectory-resolved load factor (factor (4) of the diagnosis, only
      if verified): per-leg integral of clip[(SOG(t)/v_design)^3, 0.2, 1]
      over underway trajectory points replaces the diluted leg-average
      cube law (Jensen gap: mean(v^3) >= mean(v)^3). Implemented in
      02b_sea_speed_from_ais.py + 02_emission_estimation.py merge.

Version isolation: all v5 outputs go to 02_数据_output/{processed,h3_grid,
milp_results,policy_analysis}_v5. Raw data at D:\Data is READ-ONLY.
v5 reuses the v2 cleaned legs/vessels tables (module 01 output, model-free)
and re-runs every model-bearing stage.
"""

import os
from pathlib import Path

# ============================================================
# Project paths — v5 = emission accounting revision (see header).
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repository root
OUTPUT_DIR = PROJECT_ROOT / "data" / "derived"
CODE_DIR = PROJECT_ROOT / "code" / "calc_v5"
VERSION_TAG = "v5"

PROCESSED_DIR = OUTPUT_DIR / "processed_v5"
H3_DIR = OUTPUT_DIR / f"h3_grid_{VERSION_TAG}"
RESULTS_DIR = OUTPUT_DIR / f"milp_results_{VERSION_TAG}"
POLICY_DIR = OUTPUT_DIR / f"policy_analysis_{VERSION_TAG}"
for _d in (PROCESSED_DIR, H3_DIR, RESULTS_DIR, POLICY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# v2 cleaned legs/vessels (module 01 output: pure data cleaning, no model
# assumptions) — copied into processed_v5 by 01_data_loader --reuse-v2 to
# avoid re-reading the provider CSVs.
V2_PROCESSED_DIR = OUTPUT_DIR / "processed_v2"

# ============================================================
# Raw data paths (READ-ONLY — never modify)
# ============================================================
DATA_ROOT = Path(os.environ.get("GSC_RAW_DATA_ROOT", r"D:\Data"))

# Geographic information
GEO_DIR = DATA_ROOT / "船视宝地理信息数据"
PORT_CSV = GEO_DIR / "港口数据.csv"
BERTH_DIR = GEO_DIR / "港口泊位"
SEA_AREA_CSV = GEO_DIR / "海域数据.csv"
STRAIT_CSV = GEO_DIR / "海峡数据.csv"
OCEAN_SHAPEFILE_DIR = DATA_ROOT / "地理信息数据" / "全球1：100万海洋数据集" / "全球1：100万海洋数据集" / "修正海域"

# Vessel-type-specific data paths
CONTAINER_ROOT = DATA_ROOT / "航运数据库" / "集装箱运输" / "2024集装箱数据"
CONTAINER_VESSEL_CSV = CONTAINER_ROOT / "2024集装箱船舶数据.csv"
CONTAINER_LEG_DIR = CONTAINER_ROOT / "2024集装箱船舶航段数据"
CONTAINER_TRAJ_DIRS = [
    CONTAINER_ROOT / "2024集装箱轨迹数据",
    CONTAINER_ROOT / "2024集装箱轨迹数据zxy",
]

TANKER_ROOT = DATA_ROOT / "航运数据库" / "原油运输"  # placeholder
TANKER_VESSEL_CSV = TANKER_ROOT / "原油船舶数据.csv"
TANKER_LEG_DIR = TANKER_ROOT / "原油船舶航段数据"
TANKER_TRAJ_DIRS = [TANKER_ROOT / "原油轨迹数据"]

VESSEL_TYPE = "container"  # "container" or "tanker"


def get_data_paths(vessel_type: str = None) -> dict:
    """Return raw data paths for the specified vessel type."""
    vt = vessel_type or VESSEL_TYPE
    if vt == "container":
        return {
            "vessel_csv": CONTAINER_VESSEL_CSV,
            "leg_dir": CONTAINER_LEG_DIR,
            "traj_dirs": CONTAINER_TRAJ_DIRS,
            "label": "containership",
            "label_cn": "集装箱船",
        }
    elif vt == "tanker":
        return {
            "vessel_csv": TANKER_VESSEL_CSV,
            "leg_dir": TANKER_LEG_DIR,
            "traj_dirs": TANKER_TRAJ_DIRS,
            "label": "crude_oil_tanker",
            "label_cn": "原油轮",
        }
    else:
        raise ValueError(f"Unknown vessel type: {vt}")


# ============================================================
# Emission estimation parameters (IMO 4th GHG Study)
# ============================================================
EMISSION_PARAMS = {
    # Engine load factor
    "load_factor_exponent": 3.0,       # cubic law
    "load_factor_min": 0.2,            # minimum stable engine load
    "load_factor_max": 1.0,
    "mcr_fraction": 0.75,              # P_ME = 75% MCR convention

    # Ballast correction (gamma_v)
    "ballast_correction": {
        "container": 1.0,              # containerships rarely ballast
        "tanker": 0.85,                # tankers on ballast legs
    },

    # C2: auxiliary engine + boiler share of main-engine output
    # (IMO 4th GHG Study containerships; v2-v4 used 0.0)
    "aux_engine_fraction": 0.15,

    # C3: port maneuvering / anchorage uplift outside the leg table
    "port_time_uplift": 0.05,

    # C4: trajectory-resolved load factor (segmented cube law).
    # If True, 02 uses per-leg ∫clip[(SOG/v_design)^3]dt from 02b; legs
    # without a matched trajectory fall back to the leg-average form.
    "use_traj_load_factor": True,

    # C1: WtW emission factors (g CO2e/kWh) — CORRECTED.
    # EF_kWh = EF_MJ x 3.6/eta with uniform eta = 0.50 (=> x 7.2).
    #   HFO           87.6 x 7.2 = 630.7   (v2-v4 wrong value: 315.4)
    #   MDO/MGO       89.0 x 7.2 = 640.8   (                320.4)
    #   LNG-HP        75.2 x 7.2 = 541.4   (                270.7)
    #   LNG-LP mid    86.75x 7.2 = 624.6   (                312.3)
    #   green MeOH    15.8 x 7.2 = 113.8   (                 56.9)
    #   green NH3      5.3 x 7.2 =  38.2   (                 19.1)
    # Paper Table 2, MJ column unchanged; kWh column = MJ x 7.2.
    "ef_eta": 0.50,
    "ef_wtw": {
        "HFO": 630.7,
        "MDO_MGO": 640.8,
        "LNG_HP": 541.4,              # high-pressure injection
        "LNG_LP": 624.6,              # low-pressure dual-fuel (midpoint)
        "green_methanol": 113.8,
        "green_ammonia": 38.2,
    },

    # Fuel range (nautical miles)
    "fuel_range_nm": {
        "LNG": float("inf"),
        "green_methanol": 8000,
        "green_ammonia": 5000,
    },

    # Distance correction — regionalized detour factors (v2)
    # Used only when the provider-reported sail distance is missing.
    "gc_detour_factor": 1.03,          # default (other regions)
    "detour_regions": [
        # North Atlantic (weather routing heavy)
        {"name": "north_atlantic", "lon": (-80, 10), "lat": (30, 60), "factor": 1.05},
        # Trans-Pacific (long great-circle, moderate detour)
        {"name": "trans_pacific", "lon": (120, 250), "lat": (-60, 60), "factor": 1.02},
        # Indian Ocean
        {"name": "indian_ocean", "lon": (20, 110), "lat": (-40, 30), "factor": 1.04},
    ],
    "ais_distance_threshold_km": 1000, # below this prefer AIS-measured distance

    # --- Fuel economy parameters for break-even carbon price (Eq. 28) ---
    # Prices in USD/t; LHV in MJ/kg.
    "fuel_prices_usd_t": {"HFO": 600.0, "green_methanol": 1200.0, "green_ammonia": 1000.0},
    "fuel_lhv_mj_kg": {"HFO": 40.5, "green_methanol": 19.9, "green_ammonia": 18.6},
    "infra_cost_sharing": 0.5,         # corridor share of endpoint/intermediate costs
}

# ============================================================
# H3 parameters
# ============================================================
H3_PARAMS = {
    "res_ocean": 5,          # ~33 km edge, 253 km² area
    "res_chokepoint": 7,     # ~5 km edge, 5.2 km² area
    "chokepoint_buffer_cells": 2,
    "chokepoint_regions": {
        "suez": {"lat_range": (29.5, 31.0), "lon_range": (32.0, 34.5)},
        "bab_el_mandeb": {"lat_range": (11.5, 13.5), "lon_range": (42.5, 44.5)},
        "malacca": {"lat_range": (1.0, 6.0), "lon_range": (99.0, 104.0)},
        "hormuz": {"lat_range": (25.5, 27.5), "lon_range": (55.5, 57.5)},
        "cape_of_good_hope": {"lat_range": (-35.5, -33.5), "lon_range": (17.5, 19.5)},
    },
    # Segment sampling step (km) for within-cell length attribution:
    # fine enough not to skip any traversed cell (Res5 edge ~33 km, Res7 ~2 km)
    "segment_sample_km_ocean": 5.0,
    "segment_sample_km_chokepoint": 1.0,
    # GC fallback interpolation step for legs without trajectory data
    "gc_interp_km": 10.0,
}

# ============================================================
# Corridor definition parameters
# ============================================================
CORRIDOR_PARAMS = {
    "min_annual_frequency": 4,    # n_min: minimum voyages/year
    "liner_vessel_types": ["container"],
    "tramp_vessel_types": ["tanker"],
    # Feasibility-mask fallback: ports with >= this many annual calls are kept
    # even if the berth file is missing (data-coverage protection)
    "berth_missing_call_threshold": 24,
    # Container berth type code in berth files
    "container_berth_type": 100,
}

# ============================================================
# Port infrastructure cost model (Eq. 17) — physical units
# ============================================================
COST_PARAMS = {
    "discount_rate": 0.07,
    "planning_horizon_years": 20,
    # Capital components (M USD)
    "C_base": 100.0,               # fixed base cost common to all ports
    "C_depth_penalty": 30.0,       # max dredging/civil premium when container
                                   # berths are few (< 3); -5 M USD per extra berth
    "C_remote_rate": 0.03,         # M USD per nm to nearest fuel-supply hub (capped)
    "C_remote_cap": 150.0,         # max remoteness premium (M USD)
    "fuel_hub_ports": ["SGSGP", "NLRTM", "USHOU", "CNSHA", "AEFUJ"],
}

# ============================================================
# MILP parameters
# ============================================================
MILP_PARAMS = {
    "budget_fractions": [0.05, 0.10, 0.20, 0.30],
    "discount_rate": 0.07,
    "planning_horizon_years": 20,
    "fuel_scenarios": ["LNG", "green_methanol", "green_ammonia"],
    "default_fuel": "green_methanol",   # reference fuel scenario throughout
    "adoption_shares_sensitivity": [0.3, 0.5, 0.7],
    "mip_gap": 0.001,
    "time_limit_s": 600,
    "ets_price_usd": 80.0,          # reference EU ETS price (USD/tCO2)
    "ets_recycle_shares": [0.0, 0.5, 1.0],
    "fueleu_forced_ports": 100,     # top-N EU ports forced open (FuelEU forcing)
    "m_ell_prune_k": 15,            # top-K intermediate candidates per corridor
}

# ============================================================
# Multi-period extension parameters (Eq. 29–36)
# ============================================================
MULTIPERIOD_PARAMS = {
    "periods": [2025, 2030, 2035, 2040, 2045, 2050],
    # Fuel availability window: set of commercially available fuels by period
    "fuel_availability": {
        "LNG": 2025,
        "green_methanol": 2030,
        "green_ammonia": 2035,
    },
    "learning_exponent_beta": 0.1,
    "retrofit_max_age": 15,
    # Budget shares across periods (sums to 1); total pool = 20% of cost pool
    "budget_shares": [0.10, 0.15, 0.20, 0.20, 0.20, 0.15],
    # Emission trajectory (IMO indicative checkpoints, relative to 2024)
    "emission_scenario": {2025: 1.0, 2030: 0.95, 2035: 0.85,
                          2040: 0.70, 2045: 0.50, 2050: 0.30},
    # Exogenous fleet adoption trajectory (fraction of corridor capacity)
    "adoption_scenario": {2025: 0.10, 2030: 0.25, 2035: 0.45,
                          2040: 0.65, 2045: 0.80, 2050: 0.90},
    # Fuel supply caps (Mt/yr) and throughput per bunkering port (Mt/yr/port)
    "fuel_supply_cap_Mt": {
        "green_methanol": {2030: 5.0, 2035: 15.0, 2040: 30.0, 2045: 50.0, 2050: 80.0},
        "green_ammonia":  {2035: 2.0, 2040: 8.0, 2045: 20.0, 2050: 40.0},
    },
    "fuel_throughput_Mt_per_port": {
        "green_methanol": 0.5,
        "green_ammonia": 0.25,
    },
    # PWL breakpoints for the learning-curve linearization (N_cum in ports)
    "learning_pwl_points": [0, 100, 300, 600, 1000, 1500, 2200, 3000, 4000, 5000],
}

# ============================================================
# Processing parameters
# ============================================================
PROCESSING_PARAMS = {
    "jump_speed_threshold_kn": 40,     # discard points implying > 40 kn
    "stationary_sog_threshold_kn": 0.5,
    "stationary_duration_hours": 2,
    "gap_interpolation_minutes": 30,   # interpolate if gap > 30 min
    "interpolation_step_minutes": 5,
    "n_workers": 10,                   # parallel processing workers (12 logical cores)
    "chunk_size": 500,                 # vessels per processing chunk
}

# ---------------------------------------------------------------------------
# Repository portability note (peer-review archive, 2026-10):
#   OUTPUT_DIR points at the derived tables shipped in data/derived/, so every
#   MILP / policy module (05, 05b-05e, 06, 08, 09, 12-19) runs from the
#   archived data without the licensed raw AIS export. Stage 01-03f modules
#   read the raw provider data under DATA_ROOT; set the environment variable
#   GSC_RAW_DATA_ROOT to that folder (see README.md). config.py is the ONLY
#   file whose path constants were changed when archiving the pipeline; the
#   module logic is untouched.
# ---------------------------------------------------------------------------
