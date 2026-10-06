# Competing Rules, Shared Seas — Analysis Code & Derived Data

Peer-review archive accompanying the manuscript **“Competing Rules, Shared Seas:
The Geopolitics of Green Shipping Corridor Investment”** (submitted to *Earth’s
Future*, manuscript #2026EF010217).

This package contains everything needed to audit the paper outside the
licensed raw AIS feed: the complete analysis pipeline (emission estimation,
H3 attribution, the rule-competition MILP family, all policy experiments), the
derived data tables every reported number is computed from, the institutional
country lists used by the governance classifier, and the 19 article figures.

**Raw AIS data.** The 2024 AIS trajectories, port calls, vessel particulars and
berth inventory were licensed from the Chuanshibao (Vessel Value Visualization)
maritime big-data platform of COSCO Shipping Technology Co., Ltd.
(<https://www.myvessel.cn>) under a commercial licence that precludes
redistribution; they are therefore **not** included here. Researchers should
request access from the provider. All derived quantities archived below were
generated from those licensed data and are redistributable under CC BY 4.0.

**Verified reproduction.** Re-running the global-planner MILP (20% budget,
green methanol) from the archived tables reproduces the paper’s solution
exactly — see [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Repository layout

```
├── code/
│   ├── calc_v5/           34-stage analysis pipeline (Python, Gurobi/CBC)
│   └── plot_v8/           figure scripts (matplotlib + cartopy)
├── data/
│   ├── derived/           all derived tables (parquet/csv/json) by stage
│   ├── institutional_lists/   EU/EEA and BRI country lists (with sources)
│   └── README.md          data dictionary + provenance for every file
├── figures/
│   ├── final/             the 19 article figures (Fig_01–Fig_19 + PNG source)
│   └── README.md          figure ↔ script ↔ data mapping
├── docs/REPRODUCE.md      pipeline order, stages, and reproduction protocol
├── requirements.txt       Python dependencies (pinned to the versions used)
├── LICENSE                MIT (code); data under CC BY 4.0
└── CITATION.cff           how to cite this archive
```

## Quick start

```bash
# 1. Environment (Python 3.12; the pipeline also runs on 3.10–3.13)
pip install -r requirements.txt
#    The MILP stage uses Gurobi through PuLP and falls back to the bundled
#    CBC solver when no Gurobi licence is available; both give the archived
#    optimum. COPT (Cardinal Optimizer) also works if installed.

# 2. Optional: point the pipeline at a licensed raw-data export
#    (only needed to re-run stages 01–03; see docs/REPRODUCE.md)
export GSC_RAW_DATA_ROOT=/path/to/raw/provider/export

# 3. Re-run any model stage from the archived derived data
python code/calc_v5/05_milp_solving.py --vessel-type container      # global planner + Pareto frontier
python code/calc_v5/05e_imo_carbon_milp.py --vessel-type container  # IMO carbon-price experiment
python code/calc_v5/13_sensitivity.py --vessel-type container       # sensitivity suite
python code/calc_v5/19_decomposition_2x2.py --vessel-type container # 2×2 decomposition
python code/calc_v5/06_diagnostics.py --vessel-type container       # DAI/PGI/CCI/SRI

# 4. Re-render the figures (writes to figures/)
python code/plot_v8/run_v8.py
python code/plot_v8/refine_v2_figures.py        # the remaining map/mech figures
```

All model stages read `data/derived/**` and write their outputs back next to
the archived tables, so a fresh run can be compared against the archived
file of the same name.

## Pipeline overview

`run_pipeline.py` in `code/calc_v5/` runs the full chain in dependency order.
Stage inputs fall into two groups:

| Stages | Steps | Input | Notes |
|---|---|---|---|
| A (licensed data) | 01, 02b, 02, 03, 03f, 03e, 03b, 03c | raw AIS export | needs `GSC_RAW_DATA_ROOT` and the Chuanshibao licence |
| B (archived data) | 04, 08, 05, 05b–05e, 06, 09, 12–19 | `data/derived/**` | runs as shipped; this is the peer-review path |

Stages in B regenerate every quantitative result in the paper from the
archived tables. Stage A rebuilds the derived tables from raw AIS for users
holding the licence. The full step-by-step protocol is in
[docs/REPRODUCE.md](docs/REPRODUCE.md).

## Key derived tables

| Table | Rows | What it holds |
|---|---|---|
| `data/derived/processed_v5/corridors_container.parquet` | 7,769 | corridor table: emission weights `E_e`/`W_e`, distance, speed, load factor, adoption share `alpha_e`, range flags |
| `data/derived/processed_v5/candidate_ports_container.parquet` | 1,051 | candidate-port set with annualized infrastructure cost `cost_p` and feasibility mask |
| `data/derived/processed_v5/M_ell_container.parquet` | 7,398 | corridor–intermediate-port table with leg distances |
| `data/derived/h3_grid_v5/h3_emission_grid_container.parquet` | 911,044 | cell-level H3 emission grid after the inland-cell cleanup (216.23 Mt conserved; resolution 5: 875,923 cells, resolution 7: 35,121) |
| `data/derived/policy_analysis_v5/governance_corridors_container.parquet` | 7,769 | corridor-level governance labels (EU ETS / BRI / FuelEU / vacuum) |
| `data/derived/milp_results_v5/selected_ports_container.csv` | 211 | global-planner solution at the 20% reference budget |
| `data/derived/policy_analysis_v5/breakeven_dualtrack_container.csv` | 7,769 | per-corridor break-even carbon prices, standalone and allocated tracks |

A complete dictionary with column descriptions and provenance for every
archived file is in [data/README.md](data/README.md).

## Institutional lists

`data/institutional_lists/` ships the exact country sets used by the
governance classifier (`EU/EEA member states`, the `59 Belt and Road maritime
partner countries`), as hard-coded in the pipeline, with the official sources
they were compiled from. See the README in that folder.

## Software

Python 3.12 with pandas/numpy/pyarrow, h3 4.x, scipy, matplotlib + cartopy
(map rendering), and PuLP (MILP front end; Gurobi 11 or CBC). Gurobi is a
commercial product; the pipeline falls back to CBC automatically and the
archived optima were verified to be identical under both solvers. Exact
dependency versions are in `requirements.txt`.

## Licence

Code: MIT (see `LICENSE`). Derived data: CC BY 4.0. Figures: CC BY 4.0
(reproduce with attribution of this archive and the manuscript). The licensed
raw AIS data are not covered by these terms; access them from the provider.

## Citation

If you use this code or data, please cite the manuscript and this archive:

> Wang, M. (2026). *Competing Rules, Shared Seas: The Geopolitics of Green
> Shipping Corridor Investment.* Earth’s Future (manuscript #2026EF010217).
> Analysis code and derived data: peer-review archive, `CITATION.cff`.

## Contact

Ming Wang (corresponding author), School of Navigation, Jimei University,
Xiamen 361021, China — mingw1998@outlook.com
