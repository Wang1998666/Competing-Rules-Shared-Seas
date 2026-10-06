"""
Figure Refinement — Competition Utility Matrix Heat Map (Fig U)
================================================================
Plots the full 3x3 rule-competition utility matrix U = [W^v(y^u)]
computed by calc_v4/14_utility_matrix.py:

    rows    = investing rule  u in {M^G, M^EU, M^BRI}
    columns = evaluating scope v in {Global, EU ETS, BRI}

Diagonal cells are each rule's self-best abatement; off-diagonal cells
expose the value of one rule's investment plan under another rule's
objective. MPC (global scope) is annotated on the first column.

Source: 02_数据_output/policy_analysis_v5/utility_matrix_container.csv
Output: 04_图表_figures/v7_revision/fig_U_utility_matrix_container.png
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repository root
POLICY_DIR = PROJECT_ROOT / "02_数据_output" / "policy_analysis_v5"
REFINE_FIG_DIR = PROJECT_ROOT / "figures"
REFINE_FIG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("refine_utility")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "mathtext.fontset": "stix",
})

RULES = ["M^G", "M^EU", "M^BRI"]
RULE_LABELS = ["M$^G$ (global planner)", "M$^{EU}$ (EU rule sphere)",
               "M$^{BRI}$ (BRI investment)"]
SCOPES = ["Global", "EU ETS scope", "BRI scope"]


def main(vessel_type: str = "container"):
    logger.info("[Fig U] Competition Utility Matrix")
    mat = pd.read_csv(POLICY_DIR / f"utility_matrix_{vessel_type}.csv")
    with open(POLICY_DIR / f"utility_matrix_{vessel_type}.json") as f:
        meta = json.load(f)

    data = mat.set_index("investing_rule")[["W_G_Mt", "W_EU_Mt", "W_BRI_Mt"]].values

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    im = ax.imshow(data, cmap="YlGnBu", aspect="auto",
                   vmin=0, vmax=data[0, 0] * 1.05)

    for i in range(3):
        for j in range(3):
            val = data[i, j]
            # annotate value; diagonal = self-best (bold), column 0 carries MPC
            weight = "bold" if i == j else "normal"
            color = "white" if val > data[0, 0] * 0.55 else "black"
            txt = f"{val:.2f}"
            if j == 0 and i > 0:
                mpc = mat.loc[mat["investing_rule"] == RULES[i], "MPC_pct"].iloc[0]
                txt += f"\nMPC {mpc:.1f}%"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10.5,
                    fontweight=weight, color=color, linespacing=1.4)

    # diagonal frame
    for k in range(3):
        ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False,
                                   edgecolor="#B8860B", linewidth=2.2))

    ax.set_xticks(range(3))
    ax.set_xticklabels(SCOPES, fontsize=10.5)
    ax.set_yticks(range(3))
    ax.set_yticklabels(RULE_LABELS, fontsize=10.5)
    ax.set_xlabel("Evaluating rule's abatement scope  $W^v(\\mathbf{y}^{u})$", fontsize=11)
    ax.set_ylabel("Investing rule $u$ (its own optimal plan)", fontsize=11)

    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Abatement (Mt CO$_2$e yr$^{-1}$)", fontsize=10)

    # footnote: budget / fuel basis
    ax.text(0.0, -0.16,
            f"20% budget, green-methanol scenario; activated corridors: "
            f"M$^G$ {meta['n_corridors']['G']}, "
            f"M$^{{EU}}$ {meta['n_corridors']['EU']}, "
            f"M$^{{BRI}}$ {meta['n_corridors']['BRI']}",
            transform=ax.transAxes, fontsize=8.5, color="#555555")

    plt.tight_layout()
    out = REFINE_FIG_DIR / f"fig_U_utility_matrix_{vessel_type}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {out.name}")


if __name__ == "__main__":
    main()


