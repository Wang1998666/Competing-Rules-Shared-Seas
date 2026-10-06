# Figures

`final/` contains the 19 figures of the manuscript exactly as they appear in
the article (`Fig_01`–`Fig_19`), plus the PNG source of the framework figure
(`fig1_framework_container.png`) whose PDF version is `Fig_01.pdf`. All figures
are released under CC BY 4.0.

Re-running the scripts in `code/plot_v8/` regenerates every figure from
`data/derived/**` and writes it to `figures/` (the constants were repointed
from the original output folder; script logic is unchanged).

| Manuscript figure | Archived file | Generator (function in `code/plot_v8/`) | Main data inputs |
|---|---|---|---|
| Fig. 1 — framework | `Fig_01.pdf` | `refine_v5_figures.py` → `fig1_framework` | — |
| Fig. 2 — H3 emission heatmap | `Fig_02.png` = `fig1_h3_heatmap_container.png` | `refine_v2_figures.py` → `refine_fig1_h3_heatmap` | `h3_emission_grid_container.parquet` |
| Fig. 3 — EU ETS coverage | `Fig_03.png` = `fig7_eu_ets_h3_container.png` | `refine_v5_figures.py` → `fig7_eu_ets` | `h3_eu_ets_coverage_container.parquet` |
| Fig. 4 — FuelEU exposure | `Fig_04.png` = `fig11_fueleu_container.png` | `refine_v5_figures.py` → `fig11_fueleu` | `fueleu_pressure_container.parquet` (omitted from archive; regenerate with `09_fueleu_analysis.py`) |
| Fig. 5 — BRI corridor map | `Fig_05.png` = `fig10_bri_container.png` | `refine_v5_figures.py` → `fig10_bri` | `governance_corridors`, `bri_port_summary`, MILP results |
| Fig. 6 — governance map | `Fig_06.png` = `fig_G1_governance_container.png` | `refine_v2_gov.py` → `fig_G1` | `governance_corridors`, `governance_ports` |
| Fig. 7 — BRI∩EU overlap | `Fig_07.png` = `fig_overlap_bri_eu_container.png` | `refine_new_figures.py` → overlap job | `governance_corridors` |
| Fig. 8 — siting map | `Fig_08.png` = `fig2_siting_map_container.png` | `refine_v2_figures.py` → `refine_fig2_siting_map` | `candidate_ports`, `selected_ports` |
| Fig. 9 — Pareto frontier | `Fig_09.png` = `fig3_pareto_container.png` | `refine_v5_figures.py` → `fig3_pareto` | `pareto_green_methanol_container.csv` |
| Fig. 10 — policy matrix | `Fig_10.png` = `fig_D_policy_matrix_container.png` | `refine_v2_figures.py` → `refine_fig_D` | `camp_control`, `breakeven_*` |
| Fig. 11 — utility matrix | `Fig_11.png` = `fig_U_utility_matrix_container.png` | `refine_v2_utility.py` | `utility_matrix_container.json/csv` |
| Fig. 12 — siting comparison | `Fig_12.png` = `fig_siting_comparison_container.png` | `refine_new_figures.py` → `fig_siting_comparison` | `selected_ports`, `bri_view_selected_ports_*` |
| Fig. 13 — ETS recycling | `Fig_13.png` = `fig_ets_recycling_container.png` | `refine_new_figures.py` → `fig_ets_recycling` | `sensitivity_ets_price`, fueleu comparison |
| Fig. 14 — rule-vacuum map | `Fig_14.png` = `fig9_desert_map_container.png` | `refine_v2_figures.py` → `refine_fig9_desert` | `vacuum_corridors_container.csv` |
| Fig. 15 — efficiency–equity frontier | `Fig_15.png` = `fig_E_equity_container.png` | `refine_v5_figures.py` → `fig_E_equity` | `moo_pareto_container_*.csv` |
| Fig. 16 — hub control | `Fig_16.png` = `fig_G2_control_container.png` | `refine_v2_gov.py` → `fig_G2` | `governance_ports` (HCS) |
| Fig. 17 — carbon cascade | `Fig_17.png` = `fig_C_carbon_cascade_container.png` | `refine_v5_figures.py` → `fig_C_carbon_cascade` | `breakeven_dualtrack_container.csv` |
| Fig. 18 — IMO carbon price | `Fig_18.png` = `fig_imo_carbon_container.png` | `refine_v5_figures.py` → `fig_imo_carbon` | `imo_carbon_activated_*` |
| Fig. 19 — multi-period | `Fig_19.png` = `fig8_multiperiod_container.png` | `refine_v2_figures.py` → `refine_fig8_multiperiod` | `multiperiod_*_container.csv` |

Run order used in production (reproduces the archived files):

```bash
python code/plot_v8/refine_v2_figures.py     # Figs. 2, 3, 8, 9, 10, 14, 19 and companions
python code/plot_v8/refine_v2_gov.py         # Figs. 6, 16
python code/plot_v8/refine_v2_utility.py     # Fig. 11
python code/plot_v8/refine_new_figures.py    # Figs. 7, 12, 13
python code/plot_v8/refine_v5_figures.py     # Figs. 1, 3(v2), 5, 9(v2), 15, 17, 18
python code/plot_v8/run_v8.py                # final-layout revisions of the figures revised last
```

Note: `refine_v2_abatement.py` generates supplementary abatement-scenario
figures (`fig_abate_*`) that are not part of the manuscript’s 19 figures.
