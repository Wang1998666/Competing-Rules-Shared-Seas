# Reproduction protocol

This document reproduces every quantitative result in *Competing Rules, Shared
Seas* from this archive. Stage A requires the licensed raw AIS export; **stage
B needs no raw data** and is the peer-review path.

## Stage A — rebuild the derived tables from licensed raw data (optional)

Applies only if you hold a Chuanshibao licence and a local export of the
provider data. Lay the export out as below and export one environment
variable; the module `01_data_loader.py` then reads it directly.

```text
$GSC_RAW_DATA_ROOT/
├── 船视宝地理信息数据/           # ports, berths, sea areas, straits
│   ├── 港口数据.csv
│   └── 港口泊位/
└── 航运数据库/集装箱运输/2024集装箱数据/
    ├── 2024集装箱船舶数据.csv            # vessel particulars
    ├── 2024集装箱船舶航段数据/           # legs
    └── 2024集装箱轨迹数据/, ...zxy/      # AIS trajectories
```

```bash
export GSC_RAW_DATA_ROOT=/path/to/export
cd code/calc_v5
python 01_data_loader.py --vessel-type container        # cleaning; model-free
python 02b_sea_speed_from_ais.py --vessel-type container
python 02_emission_estimation.py --vessel-type container
python 03_h3_attribution.py --vessel-type container
python 04_corridor_construction.py --vessel-type container
python 03e_clean_grid_cells.py --vessel-type container  # inland-cell cleanup, must precede 03b
python 03b_h3_corridor_paths.py --vessel-type container
python 03c_h3_policy_coverage.py --vessel-type container
```

(The production run also executed `03f_repair_bad_legs.py` before the
consolidation step of `03_h3_attribution.py` to drop physically implausible
legs; see the module docstring.)

## Stage B — regenerate every reported result (no raw data needed)

Run from `code/calc_v5/`; each module reads `data/derived/**` and overwrites
its archived output in place, so a fresh run can be diffed against the
archived file of the same name.

```bash
cd code/calc_v5
python 04_corridor_construction.py --vessel-type container   # corridor table + port cost model
python 08_bri_analysis.py --vessel-type container             # BRI marking
python 05_milp_solving.py --vessel-type container             # global planner + Pareto frontier
python 09_fueleu_analysis.py --vessel-type container          # FuelEU experiments (also rebuilds fueleu_pressure)
python 12_governance_classification.py --vessel-type container
python 05c_bri_view_milp.py --budget-fraction 0.20 --vessel-type container
python 05c_bri_view_milp.py --budget-fraction 0.10 --vessel-type container
python 05d_multiobjective_milp.py --obj2 vac --vessel-type container
python 05d_multiobjective_milp.py --obj2 bri_share --vessel-type container
python 05b_multiperiod_milp.py --vessel-type container
python 06_breakeven_price.py --vessel-type container
python 05e_imo_carbon_milp.py --vessel-type container
python 13_sensitivity.py --vessel-type container
python 14_utility_matrix.py --vessel-type container
python 15_mc_uncertainty.py --vessel-type container
python 16_robustness_checks.py --vessel-type container
python 17_sensitivity_supp.py --vessel-type container
python 18a_breakeven_allocation.py --vessel-type container
python 18b_market_nobudget.py --vessel-type container
python 18c_bri_membership_robustness.py --vessel-type container
python 19_decomposition_2x2.py --vessel-type container
python 06_diagnostics.py --vessel-type container              # DAI/PGI/CCI/SRI
```

`run_pipeline.py` chains the full production order (01 → 06_diagnostics); with
the raw data absent, stage A modules will fail at the first provider read,
which is expected — use `--from` to skip to any stage B module.

## Verified reproduction reference

Checked during archiving (2026-10-06, Python 3.12, CBC solver via PuLP):

| Quantity | Archived | Fresh re-solve from `data/derived/**` |
|---|---|---|
| Corridors / ports / M_ell rows | 7,769 / 1,051 / 7,398 | 7,769 / 1,051 / 7,398 |
| Feasible cost pool at 20% budget | 3,070.2 M USD/yr | 3,070.2 M USD/yr |
| Global-planner objective (20%, methanol) | 100.425 Mt | 100.425 Mt (status Optimal) |
| Selected ports / activated corridors | 211 / 2,758 | 211 / 2,758 (identical corridor set) |

The manuscript’s headline statistics to spot-check after a stage B run:

- Table 1 counts: 7,256 vessels, 674,055 legs, 20,779 OD pairs, 7,769
  corridors, 206.4 Mt corridor emissions (95.4% of the 216.2 Mt leg level).
- Governance shares (Discussion): BRI 58.0%, EU instruments combined 34.7%
  (ETS 20.9%), rule vacuum 7.3%; 1,014 BRI∩ETS overlap corridors.
- Mismatch: EU-only rule forgoes 77.9% of achievable abatement, BRI-only
  57.5%; FuelEU mandate costs a further 13.8–43.0%.
- Hub control: top-20 ports 51.5%; BRI 53.4% vs EU 11.3%; 380 USD/tCO₂ cap
  activates 15–28% of the planning benchmark.

## Visualization rules (why the pipeline looks the way it does)

The archived pipeline implements the project’s visualization invariants, worth
preserving in any re-plot:

1. Corridor routing graphs are built **only** from H3 cells with
   `emission_tco2e > 0` (pure AIS network); no ocean padding or great-circle
   bridging — unreachable corridors are skipped, never spliced.
2. The emission grid contains **no inland cells** (module `03e` reallocates
   their emissions to the nearest sea cell with a cKDTree, conservatively).
3. Map paths are broken at the antimeridian (90° threshold) or drawn with
   `ccrs.Geodetic()`; direct `ax.plot(lon, lat)` is prohibited.
4. The land–sea test uses the Natural Earth 110 m land mask
   (`03_h3_attribution._get_land_mask`).

Pre-flight checks after any re-plot: inland cells = 0; top-200 corridors with
>5% land-crossing samples = 0; total emission conservation = 100%.
