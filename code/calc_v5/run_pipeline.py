"""
Pipeline v5 — Orchestrator (Emission Accounting Revision)
=========================================================
Runs the complete v5 computation pipeline in dependency order. v5
implements corrections C1-C4 of the emission-gap diagnosis (see
config.py header); all outputs are version-isolated under
02_数据_output/{processed,h3_grid,milp_results,policy_analysis}_v5.
Raw data (D:\\Data) is READ-ONLY.

Order (replicates the v4 production chain, plus the calc_v4 downstream
modules):
  01 data loader (--reuse-v2: model-free tables copied from processed_v2)
  02b trajectory LF integrals (C4)      02 emission estimation (C1-C4)
  03 H3 attribution → 04 corridors → 03e sea-cell grid cleanup (in place)
  → 03b corridor paths (on the CLEAN grid) → 03c EU ETS coverage
  08 BRI → 05 MILP → 09 FuelEU → 12 governance
  05c BRI-view ×2 → 05d multi-objective ×2 → 05b multi-period
  06 break-even → 05e IMO carbon
  13 sensitivity → 14 utility → 15 MC → 16 robustness → 17 supp
  18a allocation → 18b market → 18c BRI membership → 19 decomposition
  06 diagnostics (DAI/PGI/CCI/SRI)

Usage:
    python run_pipeline.py [--vessel-type container] [--only PREFIX]
"""

import sys
import argparse
import logging
import time
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import VESSEL_TYPE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CODE_DIR = Path(__file__).parent

# (module, args) in dependency order
STEPS = [
    ("01_data_loader.py", ["--reuse-v2"]),
    ("02b_sea_speed_from_ais.py", []),
    ("02_emission_estimation.py", []),
    ("03_h3_attribution.py", []),
    ("04_corridor_construction.py", []),
    ("03e_clean_grid_cells.py", []),   # MUST precede 03b: paths built on clean grid
    ("03b_h3_corridor_paths.py", []),
    ("03c_h3_policy_coverage.py", []),
    ("08_bri_analysis.py", []),
    ("05_milp_solving.py", []),
    ("09_fueleu_analysis.py", []),
    ("12_governance_classification.py", []),
    ("05c_bri_view_milp.py", ["--budget-fraction", "0.20"]),
    ("05c_bri_view_milp.py", ["--budget-fraction", "0.10"]),
    ("05d_multiobjective_milp.py", ["--obj2", "vac"]),
    ("05d_multiobjective_milp.py", ["--obj2", "bri_share"]),
    ("05b_multiperiod_milp.py", []),
    ("06_breakeven_price.py", []),
    ("05e_imo_carbon_milp.py", []),
    ("13_sensitivity.py", []),
    ("14_utility_matrix.py", []),
    ("15_mc_uncertainty.py", []),
    ("16_robustness_checks.py", []),
    ("17_sensitivity_supp.py", []),
    ("18a_breakeven_allocation.py", []),
    ("18b_market_nobudget.py", []),
    ("18c_bri_membership_robustness.py", []),
    ("19_decomposition_2x2.py", []),
    ("06_diagnostics.py", []),
]


def run_step(module: str, args, vessel_type: str):
    logger.info(f"\n{'=' * 70}\n>>> STEP: {module} {' '.join(args)}\n{'=' * 70}")
    cmd = [sys.executable, str(CODE_DIR / module), "--vessel-type", vessel_type] + args
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    # stream log tail for visibility
    tail = (proc.stdout + proc.stderr).strip().splitlines()
    for line in tail[-6:]:
        logger.info(f"    | {line}")
    if proc.returncode != 0:
        logger.error(f"  STEP FAILED ({module}) exit={proc.returncode}")
        for line in tail[-30:]:
            logger.error(f"    ! {line}")
        raise RuntimeError(f"Step failed: {module}")
    logger.info(f"  Step {module} OK in {time.time()-t0:.0f}s")


def main():
    parser = argparse.ArgumentParser(description="GSC pipeline v5")
    parser.add_argument("--vessel-type", default=VESSEL_TYPE)
    parser.add_argument("--only", default=None,
                        help="Run only steps whose module name starts with this")
    parser.add_argument("--from", dest="from_step", default=None,
                        help="Skip all steps before the first module starting with "
                             "this prefix (resume after partial runs)")
    args = parser.parse_args()

    t_all = time.time()
    started = args.from_step is None
    for module, margs in STEPS:
        if args.from_step and not started:
            if module.startswith(args.from_step):
                started = True
            else:
                continue
        if args.only and not module.startswith(args.only):
            continue
        run_step(module, margs, args.vessel_type)
    logger.info(f"\n✓ Pipeline v5 complete in {(time.time()-t_all)/60:.1f} min")


if __name__ == "__main__":
    main()
