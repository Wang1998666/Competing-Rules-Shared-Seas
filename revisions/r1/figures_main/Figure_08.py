"""Render Figure 8 from the financing scenarios in the study's reproducibility ZIP.

Run with Python and Matplotlib. The archive defaults to the one beside the
manuscript sources. Use --archive and --output to supply other locations.
"""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


SOURCE_DIR = Path(__file__).resolve().parent
POLICY_DIR = "EF-R1/analysis/inputs/data/derived/policy_analysis_v5"
SCENARIOS = (
    "fueleu_comparison_container.json",
    "fueleu_comparison_container_viewglobal_b20_g0.5.json",
    "fueleu_comparison_container_viewglobal_b20_g1.0.json",
)
BLUE = "#415f8e"
GOLD = "#d9a844"
GREEN = "#21685e"
RED = "#a2443f"
GRAY = "#9099a3"
INK = "#233640"


def load_scenarios(archive):
    with ZipFile(archive) as bundle:
        rows = [json.loads(bundle.read(f"{POLICY_DIR}/{name}")) for name in SCENARIOS]
    for row, share in zip(rows, (0, 0.5, 1)):
        if (row["gamma"], row["view"], row["fuel"], row["budget_fraction"]) != (
            share, "global", "green_methanol", 0.2
        ):
            raise ValueError("Unexpected financing scenario in input archive")
        if row["baseline_n_ports"] + row["new_ports_added"] - row["ports_removed"] != row["rule_n_ports"]:
            raise ValueError("Changes in port membership do not match the reported total")
    return rows


def render(rows, output):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#889498",
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(9, 3.82))
    fig.text(.078, .974, "a   Financing and global abatement", va="top", size=13, weight="bold")
    fig.text(.565, .974, "b   Shortfall and port changes", va="top", size=13, weight="bold")
    budget = fig.add_axes([.078, .566, .378, .245])
    abatement = fig.add_axes([.078, .155, .378, .244])
    shortfall = fig.add_axes([.565, .566, .39, .245])
    ports = fig.add_axes([.565, .155, .39, .244])
    x = range(len(rows))
    for ax in (budget, abatement, shortfall, ports):
        ax.set_xlim(-.55, 2.55)
        ax.set_xticks(x, [f"{r['gamma']:g}" for r in rows])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#d8dfe1", linewidth=.55)
        ax.set_axisbelow(True)
        ax.tick_params(length=2, width=.55, pad=3)
    for ax in (budget, shortfall):
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
    for ax in (abatement, ports):
        ax.set_xlabel("Financing share (γ)", labelpad=2)

    bases = [r["base_budget_MUSD_yr"] for r in rows]
    supplements = [r["ets_revenue_MUSD_yr"] for r in rows]
    base_bars = budget.bar(x, bases, width=.50, color=BLUE)
    top_bars = budget.bar(x, supplements, bottom=bases, width=.50, color=GOLD)
    budget.set_ylim(0, 5750)
    budget.set_yticks([0, 2500, 5000])
    budget.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    budget.set_title("Annual budget (million USD)", loc="left", fontsize=11, pad=3)
    budget.text(0, bases[0] / 2, f"{bases[0]:,.1f}", ha="center", va="center",
                color="white", fontsize=9.5)
    for i, (base, supplement) in enumerate(zip(bases, supplements)):
        budget.text(i, base + supplement + 110, f"+{supplement:,.0f}", ha="center", fontsize=9.5)
    budget.legend([base_bars, top_bars], ["Base budget", "Simulated supplement indexed to covered baseline emissions"],
                  loc="lower left", bbox_to_anchor=(-.014, 1.23), fontsize=8.4,
                  ncol=2, columnspacing=.85, handlelength=.95, handletextpad=.35,
                  borderpad=0, frameon=False)

    benefit = [r["rule_obj_global_Mt"] for r in rows]
    reference = rows[0]["baseline_obj_global_Mt"]
    abatement.axhline(reference, color=GRAY, linestyle=(0, (4, 2)), linewidth=1.15)
    abatement.plot(x, benefit, color=GREEN, marker="o", markersize=5, linewidth=1.5)
    for i, value in enumerate(benefit):
        offset = -12 if i == 1 else 5
        abatement.annotate(f"{value:.2f}", (i, value), xytext=(0, offset),
                           textcoords="offset points", ha="center", color=GREEN, fontsize=9.5)
    abatement.text(-.49, reference + .55, f"Global benchmark {reference:.2f}",
                   color="#5d6772", fontsize=8.8, va="bottom")
    abatement.set_ylim(81, 109)
    abatement.set_yticks([85, 95, 105])
    abatement.set_title("Annual global abatement (Mt CO₂e)", loc="left", fontsize=11, pad=3)

    loss = [r["MPC_global_pct"] for r in rows]
    shortfall.bar(x, loss, width=.46, color=RED)
    shortfall.axhline(0, color=INK, linewidth=.7)
    for i, value in enumerate(loss):
        shortfall.annotate(f"{value:.2f}%", (i, value), xytext=(0, 3 if value >= 0 else -2),
                           textcoords="offset points", ha="center",
                           va="bottom" if value >= 0 else "top", color=RED, fontsize=9.5)
    shortfall.set_ylim(-7.4, 18.2)
    shortfall.set_yticks([-5, 0, 5, 10, 15])
    shortfall.set_title("Abatement shortfall vs. global benchmark (%)", loc="left", fontsize=11, pad=3)

    added = [r["new_ports_added"] for r in rows]
    removed = [r["ports_removed"] for r in rows]
    ports.bar(x, added, width=.46, color=GREEN)
    ports.bar(x, [-v for v in removed], width=.46, color=GRAY)
    ports.axhline(0, color=INK, linewidth=.7)
    for i, (gain, drop) in enumerate(zip(added, removed)):
        ports.text(i, gain + 2.5, f"+{gain}", ha="center", va="bottom", fontsize=9.5, color=GREEN)
        ports.text(i, -drop - 4, f"−{drop}" if drop else "0", ha="center", va="top", fontsize=9.5)
    ports.set_ylim(-145, 151)
    ports.set_yticks([-100, 0, 100])
    ports.set_title("Ports added (+) or removed (−) vs. global plan", loc="left", fontsize=11, pad=3)
    fig.savefig(output, format="pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=SOURCE_DIR.parent / "EF-R1_复现材料.zip")
    parser.add_argument("--output", type=Path, default=SOURCE_DIR / "Figure_08.pdf")
    args = parser.parse_args()
    render(load_scenarios(args.archive), args.output)
