# Code

Two folders:

- `calc_v5/` — the 34-module analysis pipeline (Python 3). `run_pipeline.py`
  chains the full production order; every module also runs standalone with
  `--vessel-type container`.
- `plot_v8/` — the figure scripts (matplotlib + cartopy). Run order for the
  article’s 19 figures is in `../figures/README.md`.

The only file modified when archiving the pipeline is
`calc_v5/config.py`: its path constants were repointed to the portable
repository layout (`data/derived/**` for outputs, `GSC_RAW_DATA_ROOT` for the
licensed raw-data export). All module logic is unchanged.

## calc_v5 module map (production order)

| Module | Function |
|---|---|
| `01_data_loader.py` | AIS track / leg / vessel cleaning from the provider export (licensed stage) |
| `02b_sea_speed_from_ais.py` | per-trajectory load-factor integrals (softer cubic law) |
| `02_emission_estimation.py` | IMO Fourth GHG Study bottom-up emission estimation (C1–C3 corrections) |
| `03_h3_attribution.py` | H3 spatial attribution of leg emissions, trajectory-matched, with land-constrained great-circle fallback (licensed stage) |
| `03f_repair_bad_legs.py` | drop physically implausible legs, re-aggregate the grid (licensed stage) |
| `03e_clean_grid_cells.py` | inland-cell cleanup: reallocate to nearest sea cell, cKDTree, 100% conservation |
| `03b_h3_corridor_paths.py` | pure-AIS cell routing graph, per-origin Dijkstra, no great-circle fallback |
| `03c_h3_policy_coverage.py` | EU ETS coverage map (200 nm port buffers, 1.0/0.5 voyage weights); EU/EEA country set |
| `04_corridor_construction.py` | corridor table (f ≥ 4 yr⁻¹), adoption shares, infrastructure cost model (Eq. 17) |
| `05_milp_solving.py` | global-planner MILP + Pareto frontier (5/10/20/30% budgets; methanol & ammonia) |
| `05b_multiperiod_milp.py` | multi-period extension (2025–2050, learning curve, fuel-supply caps) |
| `05c_bri_view_milp.py` | BRI single-sphere investment MILP (10% / 20% budgets) |
| `05c_bri_view_control.py` | BRI maximal-activation control solutions |
| `05d_multiobjective_milp.py` | multi-objective frontiers: vacuum reduction / BRI share |
| `05e_imo_carbon_milp.py` | IMO carbon-price scenario MILP (380–5,000 USD/tCO₂) |
| `06_breakeven_price.py` | per-corridor break-even carbon prices (Eq. 28) |
| `06_diagnostics.py` | DAI / PGI / CCI / SRI diagnostics |
| `08_bri_analysis.py` | BRI partner-country marking; BRI country set (59 codes) |
| `09_fueleu_analysis.py` | FuelEU mandate experiments (forced/no-force, γ ∈ {0, 0.5, 1.0}) |
| `12_governance_classification.py` | corridor governance classifier (EU ETS / BRI / FuelEU / vacuum) |
| `13_sensitivity.py` | sensitivity suite: τ, ETS price, uniform cost, α_e |
| `14_utility_matrix.py` | ex-post utility matrix across the three rule spheres |
| `15_mc_uncertainty.py` | Monte-Carlo uncertainty of headline quantities |
| `16_robustness_checks.py` | definitional and methodological robustness checks |
| `17_sensitivity_supp.py` | supplementary sensitivity (berth mask, n_min, priority order, distance) |
| `18a_breakeven_allocation.py` | allocated-cost break-even track (beneficiary corridors) |
| `18b_market_nobudget.py` | unconstrained market-mechanism experiments |
| `18c_bri_membership_robustness.py` | BRI membership robustness |
| `19_decomposition_2x2.py` | completed 2×2 decomposition of the EU-rule cost |
| `_stats_for_manuscript.py`, `_extract_stats_v5.py` | helpers that extract the manuscript’s headline statistics from the outputs |

## Conventions

- All paths come from `config.py`; no module hardcodes a project path.
- Modules write into `data/derived/<stage>_v5/` and are idempotent (safe to
  re-run; outputs overwrite the archived file of the same name, which is how a
  fresh run is compared with the archived results).
- The MILP stage prefers Gurobi via PuLP and falls back to CBC automatically;
  the archived optima were verified identical under both.
