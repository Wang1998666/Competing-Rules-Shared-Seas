# Derived data — dictionary and provenance

All tables in this folder are derived from the licensed raw AIS export
(Chuanshibao platform, COSCO Shipping Technology; see the top-level
`README.md`) or from the public institutional/economic sources listed in the
manuscript’s Open Research statement. They are redistributed under CC BY 4.0.

Folder names mirror the pipeline stage outputs; `processed_v5` holds the
model-input tables (corridors, ports, intermediates), `h3_grid_v5` the spatial
attribution products, `milp_results_v5` the optimization solutions,
`policy_analysis_v5` the governance and policy-experiment outputs, and
`diagnostics_v5` the diagnostic indicators.

Large model-free intermediates are deliberately not archived (they are
regenerable from the pipeline): the leg-level voyage table and emissions
(`legs_container.parquet`, `emissions_container.parquet`,
`traj_leg_integrals_container.parquet`), the intermediate emission grid
`h3_emission_grid_int_container.parquet`, the chunked attribution parts
`parts_container/`, and the FuelEU pressure field
`fueleu_pressure_container.parquet` (44 MB). Everything needed to re-run every
reported model and every figure except `Fig_04` (which reads the FuelEU
pressure field) is present; re-running stage A or module 09 regenerates the
omitted files.

## processed_v5 — model inputs

| File | Rows | Columns | Notes |
|---|---|---|---|
| `corridors_container.parquet` | 7,769 | origin_port, dest_port, frequency, n_vessels, total_emission, mean_distance, mean_speed, mean_load_factor, EF_avg_e, n_legs_direct, corridor_id, E_e, eta_LNG/eta_green_methanol/eta_green_ammonia, alpha_e, W_e, origin_lon/lat, dest_lon/lat, gc_distance_nm, range_violating_* , k_e_* , DAI_green_ammonia/methanol, is_desert, PGI | corridor table of the paper (f ≥ 4 yr⁻¹, undirected). `E_e` = WtW emissions, `W_e` = realizable abatement (E_e · eta_e · alpha_e) |
| `candidate_ports_container.parquet` | 1,051 | port_code, port_name_cn, country_code, country_cn/cn/en, port_lon/lat, continent, n_corridors_endpoint, n_container_berths, is_feasible, is_intermediate, C_base_MUSD, C_depth_MUSD, C_remote_MUSD, C_cap_MUSD, cost_p, SRI | candidate-port set; `cost_p` = annualized capital cost (Eq. 17), `is_feasible` = berth mask (944 feasible) |
| `M_ell_container.parquet` | 7,398 | corridor_id, origin_port, dest_port, intermediate_port, fuel, d_origin_nm, d_dest_nm | feasible bunkering intermediates per corridor/fuel |
| `ports_global.parquet` | — | global port register with country codes | source of the country mapping used by the governance classifier |
| `vessels_container.parquet` | — | vessel particulars (DWT, engine power, design speed, fuel type, build year) | linked to AIS fleet via MMSI/IMO |
| `emission_stats_container.csv` | — | summary statistics of the bottom-up estimate | calibrates Table 1 of the manuscript |

## h3_grid_v5 — spatial attribution

| File | Rows | Columns | Notes |
|---|---|---|---|
| `h3_emission_grid_container.parquet` | 911,044 | h3_id, emission_tco2e, lat, lon, resolution | H3 emission grid after the inland-cell cleanup (216.23 Mt conserved; res 5: 875,923 cells, res 7: 35,121) |
| `h3_emission_grid_summary_0p5deg_container.csv` | — | lat_bin, lon_bin, emission_tco2e, n_h3_cells | convenience 0.5° aggregation of the grid for quick checks |
| `corridor_h3_paths_container.parquet` | 1,279,990 | corridor_id, h3_id, seq_order, emission_tco2e | cell-level corridor paths from the pure-AIS Dijkstra routing network |
| `h3_corridor_coverage_container.parquet` | — | cell ↔ corridor coverage products | used by the governance/PGI maps |
| `h3_eu_ets_coverage_container.parquet` | 111,191 | h3_id, lat, lon, coverage_type, weight | EU ETS coverage map (port buffers 200 nm, voyage weights 1.0/0.5) |
| `corridor_pgi_h3_container.csv` | 6,994 | corridor_id, PGI_h3, n_h3_cells, eu_covered_cells | per-corridor Policy Gap Index at H3 precision |
| `h3_stats_container.json` | — | n_cells, total_Mt, conservation_ratio, n_res5, n_res7 | grid-level checks (216.23 Mt, conservation 1.000) |

## milp_results_v5 — optimization solutions

| File | Content |
|---|---|
| `activated_corridors_container.csv` | global-planner solution at the 20% reference budget (2,758 activated corridors; objective 100.425 Mt) |
| `selected_ports_container.csv` | the 211 selected ports of that solution |
| `pareto_green_methanol_container.csv`, `pareto_green_ammonia_container.csv` | Pareto frontiers over budget fractions 5/10/20/30% for both fuels |
| `multiperiod_*.csv` (+ `_beta005`, `_beta020`, `_learn`, `_supply050`, `_supply150` variants) | multi-period extension corridors/ports/summaries 2025–2050 |

## policy_analysis_v5 — governance and policy experiments

| File | Content |
|---|---|
| `governance_corridors_container.parquet` | corridor governance labels: is_bri, is_bri_full, origin_eu/dest_eu, eu_ets_endpoint/full/share, fueleu_max/exposed, gov_type, is_vacuum |
| `governance_ports_container.parquet` | port-level camp membership (BRI / EU / Other) and hub-control share `HCS_pct` |
| `bri_*_container.*` | BRI corridor marking, port summary, membership robustness |
| `bri_view_*_b10/b20.*` | BRI-view MILP (single-sphere investment) solutions at 10% and 20% budgets |
| `bri_view_control_*` | BRI maximal-activation (control) solutions |
| `fueleu_*_container.*` | FuelEU mandate experiments: forced/no-force variants at budgets 10–20% and adoption share γ ∈ {0, 0.5, 1.0} |
| `meU_/meB_/meU_forced_*` | utility-matrix components (global vs. EU vs. BRI spheres) |
| `moo_*_container_vac/bri_share.*` | multi-objective Pareto frontiers (vacuum reduction, BRI share) |
| `breakeven_prices_container.csv`, `breakeven_dualtrack_container.csv` | per-corridor break-even carbon prices: standalone, fuel-only, and allocated tracks |
| `imo_carbon_activated_/ports_380/500/1000/2000` | IMO carbon-price experiments (380–5,000 USD/tCO₂) |
| `market_unconstrained_standalone_/allocated_*` | market-mechanism experiments under standalone and shared-cost accounting |
| `market_design_summary_container.*` | market-design summary |
| `camp_control_container.csv`, `hub_control_container.csv`, `vacuum_corridors_container.csv` | power-and-price results: camp control, top-port hub control, rule-vacuum lanes |
| `decomposition_2x2_container.json` | completed 2×2 decomposition of the EU-rule cost |
| `sensitivity_*.csv/json` | sensitivity suite (τ, ETS price, uniform cost, α_e, berth mask, distance, n_min, priority order, LNG factors) |
| `robustness_*.csv/json`, `sensitivity_supp_*` | robustness checks and supplementary sensitivity |
| `mc_uncertainty_container.json` | Monte-Carlo uncertainty summary |
| `utility_matrix_container.*` | ex-post utility matrix across the three rule spheres |

## diagnostics_v5

| File | Content |
|---|---|
| `cci_top50_container.csv` | top-50 corridors by Corridor Concentration Index |
| `diagnostics_summary_container.csv` | DAI / PGI / CCI / SRI summary |

## Regenerating omitted tables

| Omitted file | Regenerate with |
|---|---|
| `legs_container.parquet` / `emissions_container.parquet` / `traj_leg_integrals_container.parquet` | `01_data_loader.py` → `02b_sea_speed_from_ais.py` → `02_emission_estimation.py` (needs licensed raw data) |
| `h3_emission_grid_int_container.parquet` | `03_h3_attribution.py` (needs licensed raw data) |
| `parts_container/` | `03_h3_attribution.py` resumable chunks (needs licensed raw data) |
| `fueleu_pressure_container.parquet` | `09_fueleu_analysis.py` (runs from the archived derived data) |
