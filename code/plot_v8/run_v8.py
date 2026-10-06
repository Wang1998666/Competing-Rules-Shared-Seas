"""
run_v8.py — v8 revision driver
Runs only the figures revised in v8 (based on the plot_v7revision scripts,
copied into this folder; originals untouched). Outputs to
04_图表_figures/v8_revision/.

Revisions:
  fig_C_carbon_cascade   vertical-line labels 7.5 -> 10, main legend lowered
  fig_E_equity           equity-premium box moved left, (b) legend enlarged
  fig_ets_recycling      (a) legend 1x4 -> 2x2, green labels pulled inside,
                         (b) legend centered
  fig_G1_governance      map legend / bar labels / legend enlarged
  fig_siting_comparison  stats box + legends enlarged
  fig8_multiperiod       n-ports labels + figure legend enlarged
  fig3_pareto            diminishing-returns box moved down off the curves
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from refine_v5_figures import (fig_C_carbon_cascade, fig_E_equity,
                               fig3_pareto, fig8_multiperiod)
from refine_new_figures import fig_ets_recycling, fig_siting_comparison
from refine_v2_gov import fig_G1

JOBS = [
    ("fig_C_carbon_cascade", fig_C_carbon_cascade),
    ("fig_E_equity", fig_E_equity),
    ("fig_ets_recycling", fig_ets_recycling),
    ("fig_G1_governance", fig_G1),
    ("fig_siting_comparison", fig_siting_comparison),
    ("fig8_multiperiod", fig8_multiperiod),
    ("fig3_pareto", fig3_pareto),
]

if __name__ == "__main__":
    failed = []
    for name, fn in JOBS:
        try:
            fn()
        except Exception as e:
            failed.append(name)
            print(f"[FAIL] {name}: {e}", flush=True)
    print("=" * 60)
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    print("All v8 figures rendered OK.")
