"""
Module 15 (v4): Monte Carlo Uncertainty Band for the Headline Abatement
=======================================================================
Implements the uncertainty propagation promised in Section 4.1 of the
manuscript: 500 replications with the load-factor exponent drawn from
U[2.7, 3.3] and the WtW emission factor from +/-10% of its central value.

The headline figure is the M^G solution's abatement (37 Mt at the 20%
budget). Because the sensitivity analysis (Section 5.5) shows the optimal
port set is essentially invariant to parameter perturbation (port set
varies by at most two ports), the MC is evaluated CONDITIONAL ON the
optimal siting: the activated corridor set is held fixed and only the
per-corridor abatement weights W_e = E_e * eta_e * alpha_e are re-drawn.

Parameterization per replication:
  * exponent k ~ U[2.7, 3.3]:  LF' = clip(LF^(k/3), 0.2, 1.0)
                                E_e' = E_e * LF' / LF
  * WtW factor delta ~ U[0.9, 1.1] applied to the incumbent fleet factor:
                                eta' = 1 - EF_alt / (EF_avg_e * delta)
                                (clipped to [0, 1])

Outputs (02_数据_output/policy_analysis_v4/):
  mc_uncertainty_{vt}.json — mean / median / 95% CI of total abatement
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import PROCESSED_DIR, RESULTS_DIR, POLICY_DIR, VESSEL_TYPE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

N_MC = 500
RNG = np.random.default_rng(42)

EF_ALT = 56.9          # green methanol WtW factor (g CO2e/kWh), Table 2
EXPONENT_RANGE = (2.7, 3.3)
EF_DELTA_RANGE = (0.9, 1.1)


def main(vessel_type: str):
    act = pd.read_csv(RESULTS_DIR / f"activated_corridors_{vessel_type}.csv")
    cols = ["E_e", "eta_green_methanol", "alpha_e", "mean_load_factor"]
    act = act.dropna(subset=["mean_load_factor"])
    E = act["E_e"].values
    eta = act["eta_green_methanol"].values
    alpha = act["alpha_e"].values
    LF = np.clip(act["mean_load_factor"].values, 0.2, 1.0)
    logger.info(f"  Activated corridors with LF data: {len(act)}")

    totals = np.zeros(N_MC)
    for i in range(N_MC):
        k = RNG.uniform(*EXPONENT_RANGE)
        delta = RNG.uniform(*EF_DELTA_RANGE)
        lf_i = np.clip(LF ** (k / 3.0), 0.2, 1.0)
        E_i = E * lf_i / LF
        eta_i = np.clip(1.0 - EF_ALT / (EF_ALT / (1.0 - eta) * delta), 0.0, 1.0)
        totals[i] = (E_i * eta_i * alpha).sum() / 1e6

    result = {
        "n_mc": N_MC,
        "abatement_mean_Mt": round(float(totals.mean()), 2),
        "abatement_median_Mt": round(float(np.median(totals)), 2),
        "ci95_lower_Mt": round(float(np.percentile(totals, 2.5)), 2),
        "ci95_upper_Mt": round(float(np.percentile(totals, 97.5)), 2),
        "cv_pct": round(float(totals.std() / totals.mean() * 100), 2),
        "point_estimate_Mt": round(float((E * eta * alpha).sum() / 1e6), 2),
    }
    with open(POLICY_DIR / f"mc_uncertainty_{vessel_type}.json", "w") as f:
        json.dump(result, f, indent=2)
    logger.info("\n" + json.dumps(result, indent=2))
    logger.info("Saved: mc_uncertainty_* (v4)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vessel-type", default=VESSEL_TYPE)
    args = parser.parse_args()
    main(args.vessel_type)
