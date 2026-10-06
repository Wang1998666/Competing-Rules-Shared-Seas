"""Extract Table-1 statistics from v5 outputs (read-only)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repository root
OUT = PROJECT_ROOT / "02_数据_output"

stats = {}

# leg-level emissions
es = pd.read_csv(OUT / "processed_v5" / "emission_stats_container.csv").iloc[0]
stats["leg_total_Mt"] = es["total_emission_tco2e"] / 1e6
stats["n_vessels"] = int(es["n_vessels"])
stats["n_legs"] = int(es["n_legs"])
stats["n_od"] = int(es["n_od_pairs"])

# corridors
corr = pd.read_parquet(OUT / "processed_v5" / "corridors_container.parquet")
stats["n_corridors"] = len(corr)
stats["corr_total_Mt"] = corr["E_e"].sum() / 1e6
stats["corr_coverage_pct"] = 100 * corr["E_e"].sum() / es["total_emission_tco2e"]
stats["mean_freq"] = float(corr["f_od"].mean()) if "f_od" in corr.columns else None
freq_col = [c for c in corr.columns if "freq" in c.lower()]
stats["freq_cols"] = freq_col
if freq_col:
    f = corr[freq_col[0]]
    stats["mean_freq"] = float(f.mean())
    stats["median_freq"] = float(f.median())
stats["mean_Ee_t"] = float(corr["E_e"].mean())
stats["median_Ee_t"] = float(corr["E_e"].median())
dist_col = [c for c in corr.columns if "distance" in c.lower()]
if dist_col:
    d = corr[dist_col[0]]
    stats["mean_dist_km"] = float(d.mean())
    stats["median_dist_km"] = float(d.median())
stats["n_same_port"] = int((corr["origin_port"] == corr["dest_port"]).sum())

# H3 grid
grid_files = list((OUT / "h3_grid_v5").glob("*.parquet"))
stats["h5_files"] = [f.name for f in grid_files]
for f in grid_files:
    try:
        g = pd.read_parquet(f, columns=None)
        emis_col = [c for c in g.columns if "emis" in c.lower()]
        if emis_col:
            e = g[emis_col[0]]
            stats[f"grid_{f.stem}_ncells"] = len(g)
            stats[f"grid_{f.stem}_total_Mt"] = float(e.sum() / 1e6)
            stats[f"grid_{f.stem}_ncells_pos"] = int((e > 0).sum())
    except Exception as ex:
        stats[f"grid_{f.stem}_err"] = str(ex)

# activated corridors of M^G 20%
act = pd.read_csv(OUT / "milp_results_v5" / "activated_corridors_container.csv")
stats["MG20_n_corr"] = len(act)
sel = pd.read_csv(OUT / "milp_results_v5" / "selected_ports_container.csv")
stats["MG20_n_ports"] = len(sel)

print(json.dumps(stats, indent=2, default=str))
