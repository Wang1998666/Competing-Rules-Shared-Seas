"""Supplementary stats for the v5 manuscript update."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repository root
OUT = PROJECT_ROOT / "02_数据_output"
PROC = OUT / "processed_v5"
H3 = OUT / "h3_grid_v5"
RES = OUT / "milp_results_v5"
POL = OUT / "policy_analysis_v5"
VT = "container"

out = {}

# ---- 1. Desert (both calibers) ----
h3_df = pd.read_parquet(H3 / f"h3_emission_grid_{VT}.parquet")
h3_df = h3_df[h3_df["emission_tco2e"] > 0]
h3_df = h3_df[h3_df["lat"].abs() < 72]
p75 = h3_df["emission_tco2e"].quantile(0.75)
total_emission = h3_df["emission_tco2e"].sum()

coverage = pd.read_parquet(H3 / f"h3_corridor_coverage_{VT}.parquet")
covered_all = set(coverage["h3_id"])
desert_all = (h3_df["emission_tco2e"] >= p75) & (~h3_df["h3_id"].isin(covered_all))
out["desert_allcorr_Mt"] = h3_df.loc[desert_all, "emission_tco2e"].sum() / 1e6
out["desert_allcorr_pct"] = h3_df.loc[desert_all, "emission_tco2e"].sum() / total_emission * 100

# activated corridors of M^G -> their H3 paths
act = pd.read_csv(RES / f"activated_corridors_{VT}.csv")
act_ids = set(act["corridor_id"])
paths = pd.read_parquet(H3 / f"corridor_h3_paths_{VT}.parquet")
act_paths = paths[paths["corridor_id"].isin(act_ids)]
covered_act = set(act_paths["h3_id"])
desert_act = (h3_df["emission_tco2e"] >= p75) & (~h3_df["h3_id"].isin(covered_act))
out["desert_actcorr_Mt"] = h3_df.loc[desert_act, "emission_tco2e"].sum() / 1e6
out["desert_actcorr_pct"] = h3_df.loc[desert_act, "emission_tco2e"].sum() / total_emission * 100
out["h3_total_Mt_lat72"] = total_emission / 1e6
out["h3_p75_t"] = p75
out["n_cells_p75plus"] = int(desert_all.sum() + (h3_df["emission_tco2e"] >= p75).sum() - (h3_df["emission_tco2e"] >= p75).sum())  # placeholder
out["n_cells_ge_p75"] = int((h3_df["emission_tco2e"] >= p75).sum())

# ---- 2. Fig D policy matrix ----
corridors = pd.read_parquet(PROC / f"corridors_{VT}.parquet")
bri_df = pd.read_parquet(POL / f"bri_corridors_{VT}.parquet")
corridors = corridors.merge(bri_df[["corridor_id", "is_bri"]], on="corridor_id", how="left")
corridors["is_bri"] = corridors["is_bri"].fillna(False)
corridors["milp_activated"] = corridors["corridor_id"].isin(act_ids)
fueleu = pd.read_csv(POL / f"fueleu_activated_corridors_{VT}.csv")
corridors["fueleu_activated"] = corridors["corridor_id"].isin(set(fueleu["corridor_id"]))
corridors["eu_connected"] = corridors["mean_distance"] < 3000
bp = pd.read_csv(POL / f"breakeven_prices_{VT}.csv").set_index("corridor_id")["p_star_usd"]
corridors["p_star"] = corridors["corridor_id"].map(bp)
corridors["carbon_viable"] = corridors["p_star"] < 5000
corridors["cape_rerouted"] = corridors["mean_distance"] > 12000

categories = {
    "BRI": corridors["is_bri"],
    "Non-BRI": ~corridors["is_bri"],
    "EU-Connected": corridors["eu_connected"],
    "Long-haul": corridors["cape_rerouted"],
    "Short-sea": corridors["mean_distance"] < 3000,
    "High-E": corridors["E_e"] >= corridors["E_e"].quantile(0.8),
}
dims = {
    "MILP": "milp_activated",
    "FuelEU": "fueleu_activated",
    "BRI_member": "is_bri",
    "CarbonViable": "carbon_viable",
    "EUConn": "eu_connected",
}
matrix = {}
for cname, cmask in categories.items():
    sub = corridors[cmask]
    matrix[cname] = {dname: round(sub[dcol].sum() / len(sub) * 100, 1)
                     for dname, dcol in dims.items()}
    matrix[cname]["n"] = int(len(sub))
out["policy_matrix"] = matrix

# ---- 3. Methanol price 700 USD/t sensitivity ----
dt = pd.read_csv(POL / f"breakeven_dualtrack_{VT}.csv")
# caliber = the 6,548 corridors with a defined break-even price (dE_e > 0),
# matching breakeven_dualtrack_summary tracks section and Fig. C
d = dt[np.isfinite(dt["p_star_usd"]) & (dt["dE_e"] > 0)].copy()
out["n_breakeven_corridors"] = int(len(d))
ratio = (700 / 19.9 - 600 / 40.5) / (1200 / 19.9 - 600 / 40.5)
out["dp_ratio_700"] = ratio
for label, infra_col in [
    ("fuel_only", None),
    ("allocated", "C_infra_alloc_usd"),
    ("standalone", "C_infra_usd"),
]:
    if infra_col is None:
        p = d["dC_fuel_usd"] * ratio / d["dE_e"]
    else:
        p = (d["dC_fuel_usd"] * ratio + d[infra_col]) / d["dE_e"]
    out[f"share_le380_at700_{label}"] = float((p <= 380).mean() * 100)
    out[f"median_at700_{label}"] = float(p.median())
# reference-price shares for context
for label, col in [("fuel_only", "p_fuel_only_usd"),
                   ("allocated", "p_star_alloc_usd"),
                   ("standalone", "p_star_usd")]:
    out[f"share_le380_ref_{label}"] = float((d[col] <= 380).mean() * 100)

# analytic fuel-only floor (v5 EFs): premium 45.49 USD/GJ, EF_HFO 630.7 g/kWh
prem = 1200 / 19.9 - 600 / 40.5
ef_hfo, ef_meoh = 630.7, 113.8
eta_max = 1 - ef_meoh / ef_hfo
out["floor_full_abate_usd"] = float(prem * 3.6 / (ef_hfo / 1000))
out["floor_eta_max_usd"] = float(prem * 3.6 / (ef_hfo / 1000) / eta_max)
out["eta_max"] = float(eta_max)
# methanol price at which floor(eta_max) = 380
p_meoh_380 = (380 / (3.6 / (ef_hfo / 1000) / eta_max) + 600 / 40.5) * 19.9
out["meoh_price_floor380_usd_t"] = float(p_meoh_380)

# ---- 4. Forced EU ports budget share ----
ports = pd.read_parquet(PROC / f"candidate_ports_{VT}.parquet").drop_duplicates("port_code")
ports_f = ports[ports["is_feasible"]].copy()
cost_p = ports_f.set_index("port_code")["cost_p"]
base_budget = cost_p.sum() * 0.2
sys.path.insert(0, str(PROJECT_ROOT / "03_代码_pipeline" / "calc_v5"))
from importlib import import_module
_03c = import_module("03c_h3_policy_coverage")
eu_codes = set(ports_f[ports_f["country_code"].isin(_03c.EU_EEA_COUNTRIES)]["port_code"])
corr_eu_count = pd.concat([
    corridors.loc[corridors["origin_port"].isin(eu_codes), "origin_port"],
    corridors.loc[corridors["dest_port"].isin(eu_codes), "dest_port"],
]).value_counts()
ranked = [c for c in corr_eu_count.index if c in cost_p.index]
cum = np.cumsum([cost_p[p] for p in ranked[:100]])
out["forced100_cost_MUSD"] = float(cum[-1])
out["forced100_budget_share_pct"] = float(cum[-1] / base_budget * 100)
out["base_budget_MUSD"] = float(base_budget)

# ---- 5. Multiperiod 2030 checkpoint detail ----
mp30 = pd.read_csv(RES / f"multiperiod_corridors_2030_{VT}.csv")
same_port = (mp30["origin_port"] == mp30["dest_port"]).sum()
out["mp2030_n_corr"] = int(len(mp30))
out["mp2030_same_port"] = int(same_port)
out["mp2030_interport"] = int(len(mp30) - same_port)

with open(OUT / "v5_manuscript_stats.json", "w") as f:
    json.dump(out, f, indent=2, default=float)
print(json.dumps(out, indent=2, default=float))
