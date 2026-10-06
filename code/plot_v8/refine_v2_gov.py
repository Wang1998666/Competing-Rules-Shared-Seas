"""
Figure Refinement — New Governance Figures (refine_all_figures style)
=======================================================================
Code-version style (figure_refine/code, academic publication style):
  - Times New Roman + STIX math, no main titles, (a)/(b) panel labels
  - Outputs to figure_refine/figures/ (same directory as refine_all_figures)

Figures:
  Fig G1: Governance geography of corridors (RQ1)
  Fig G2: Camp control over abatement (RQ3)
  Fig E : Real efficiency-equity frontier from 05d MOO (RQ2)
          [OVERWRITES the simulated version from refine_all_figures]
  Fig 1 : Framework flowchart

Sources adapted from code/13_governance_visualization.py,
code/14_framework_figure.py, code/11_deep_analysis_viz.py (Fig E).
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import h3

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repository root
CODE_DIR = PROJECT_ROOT / "code" / "calc_v5"
OUTPUT_DIR = PROJECT_ROOT / "data" / "derived"
REFINE_FIG_DIR = PROJECT_ROOT / "figures"
REFINE_FIG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(CODE_DIR))

PROCESSED_DIR = OUTPUT_DIR / "processed_v5"
H3_DIR = OUTPUT_DIR / "h3_grid_v5"
RESULTS_DIR = OUTPUT_DIR / "milp_results_v5"
POLICY_DIR = OUTPUT_DIR / "policy_analysis_v5"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("refine_gov")

# ─── Academic style (same as refine_all_figures) ─────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "mathtext.fontset": "stix",
})

OCEAN_COLOR = "#dceef7"
LAND_COLOR = "#f0ede8"
COAST_COLOR = "#999999"

GOV_COLORS = {
    "EU_ETS": "#4169E1",
    "BRI_only": "#DAA520",
    "FuelEU_exposed": "#8E44AD",
    "Vacuum": "#E74C3C",
}
GOV_ORDER = ["EU_ETS", "BRI_only", "FuelEU_exposed", "Vacuum"]
CAMP_COLORS = {"BRI": "#DAA520", "EU": "#4169E1", "Other": "#8D8D8D"}

VESSEL_TYPE = "container"


def setup_basemap(fig, position=111, extent=None):
    ax = fig.add_subplot(position, projection=ccrs.PlateCarree())
    if extent is None:
        extent = [-180, 180, -56, 72]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '110m',
                   facecolor=OCEAN_COLOR), zorder=0)
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m',
                   facecolor=LAND_COLOR), zorder=0)
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'coastline', '110m',
                   facecolor='none', edgecolor=COAST_COLOR, linewidth=0.4), zorder=1)
    ax.add_feature(cfeature.NaturalEarthFeature(
        'cultural', 'admin_0_boundary_lines_land', '110m',
        facecolor='none', edgecolor="#b0a89a", linewidth=0.35,
        linestyle=(0, (3, 2))), zorder=1)
    gl = ax.gridlines(draw_labels=False, linestyle=":", linewidth=0.45,
                      color="#8fa3b8", alpha=0.45)
    gl.set_zorder(1)
    return ax


def panel_label(ax, text, x=0.015, y=0.97):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left")


def split_at_antimeridian(lons, lats, threshold=90):
    segments = []
    seg_lons, seg_lats = [lons[0]], [lats[0]]
    for i in range(1, len(lons)):
        if abs(lons[i] - lons[i - 1]) > threshold:
            if len(seg_lons) >= 2:
                segments.append((np.array(seg_lons), np.array(seg_lats)))
            seg_lons, seg_lats = [lons[i]], [lats[i]]
        else:
            seg_lons.append(lons[i])
            seg_lats.append(lats[i])
    if len(seg_lons) >= 2:
        segments.append((np.array(seg_lons), np.array(seg_lats)))
    return segments


def save(name):
    path = REFINE_FIG_DIR / f"{name}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {path.name}")


# ============================================================
# Fig G1: Governance geography
# ============================================================

def fig_G1():
    logger.info("[Fig G1] Governance Geography")
    gov_file = POLICY_DIR / f"governance_corridors_{VESSEL_TYPE}.parquet"
    paths_file = H3_DIR / f"corridor_h3_paths_{VESSEL_TYPE}.parquet"
    if not gov_file.exists():
        logger.warning("  governance_corridors missing — run 12 first")
        return

    gov = pd.read_parquet(gov_file)
    paths = pd.read_parquet(paths_file) if paths_file.exists() else None
    ports = pd.read_parquet(PROCESSED_DIR / "ports_global.parquet").drop_duplicates("port_code")

    n_top = 15
    top_ids = []
    for gt in GOV_ORDER:
        top_ids.extend(gov[gov["gov_type"] == gt].nlargest(n_top, "E_e")["corridor_id"].tolist())

    fig = plt.figure(figsize=(12, 10.5))

    # (a) map
    ax = setup_basemap(fig, 211)
    if paths is not None:
        path_lookup = paths.groupby("corridor_id")["h3_id"].apply(list).to_dict()
        for cid in top_ids:
            gt = gov.loc[gov["corridor_id"] == cid, "gov_type"].iloc[0]
            cells = path_lookup.get(cid)
            if cells is None or len(cells) < 2:
                continue
            lats = [h3.cell_to_latlng(h)[0] for h in cells]
            lons = [h3.cell_to_latlng(h)[1] for h in cells]
            for sl, sa in split_at_antimeridian(lons, lats):
                ax.plot(sl, sa, color=GOV_COLORS[gt], linewidth=1.6, alpha=0.85,
                        zorder=6, transform=ccrs.Geodetic())
    ax.scatter(ports["port_lon"], ports["port_lat"], c="#555555", s=3,
               alpha=0.35, zorder=4, transform=ccrs.PlateCarree())
    patches = [mpatches.Patch(color=GOV_COLORS[gt], label=gt.replace("_", " "))
               for gt in GOV_ORDER]
    ax.legend(handles=patches, loc="lower left", fontsize=10.5, framealpha=0.9)
    panel_label(ax, "(a)")

    # (b) shares
    ax2 = fig.add_subplot(212)
    total_em = gov["E_e"].sum()
    n_corr = len(gov)
    share_em = [gov.loc[gov["gov_type"] == gt, "E_e"].sum() / total_em * 100
                for gt in GOV_ORDER]
    share_n = [gov.loc[gov["gov_type"] == gt, "corridor_id"].count() / n_corr * 100
               for gt in GOV_ORDER]
    x = np.arange(len(GOV_ORDER))
    width = 0.36
    b1 = ax2.bar(x - width / 2, share_em, width, color=[GOV_COLORS[g] for g in GOV_ORDER],
                 edgecolor="black", linewidth=0.4, label="Emission share (%)")
    b2 = ax2.bar(x + width / 2, share_n, width, color=[GOV_COLORS[g] for g in GOV_ORDER],
                 edgecolor="black", linewidth=0.4, alpha=0.45,
                 label="Corridor count share (%)")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 2), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels([g.replace("_", " ") for g in GOV_ORDER])
    ax2.set_ylabel("Share of total (%)")
    ax2.legend(fontsize=10.5, loc="upper right")
    ax2.grid(axis="y", alpha=0.3)
    panel_label(ax2, "(b)")

    plt.tight_layout()
    save(f"fig_G1_governance_{VESSEL_TYPE}")


# ============================================================
# Fig G2: Camp control over abatement
# ============================================================

def fig_G2():
    logger.info("[Fig G2] Camp Control")
    ports_file = POLICY_DIR / f"governance_ports_{VESSEL_TYPE}.parquet"
    if not ports_file.exists():
        logger.warning("  governance_ports missing — run 12 first")
        return

    pg = pd.read_parquet(ports_file)
    pg = pg[pg["unlocked_abate"] > 0].sort_values("HCS_pct", ascending=False)
    top20 = pg.head(20)

    fig = plt.figure(figsize=(12.5, 9.8))

    # (a) Top-20 hub bars — rotated 90° clockwise: vertical bars with the
    # former top bar (SGSGP) placed at the right end. Top panel.
    ax = fig.add_axes([0.07, 0.615, 0.90, 0.33])
    rev = top20.iloc[::-1]
    xpos = np.arange(len(rev))
    ax.bar(xpos, rev["HCS_pct"], color=[CAMP_COLORS[c] for c in rev["camp"]],
           edgecolor="black", linewidth=0.4)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{p} ({c})" for p, c in zip(rev["port_code"], rev["camp"])],
                       rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Hub control share HCS (%)", fontsize=11)
    ax.tick_params(axis="y", labelsize=9)
    for x, h in zip(xpos, rev["HCS_pct"]):
        ax.annotate(f"{h:.2f}", xy=(x, h), xytext=(0, 2), textcoords="offset points",
                    ha="center", fontsize=7.5)
    ax.grid(axis="y", alpha=0.3)
    camp_patches = [mpatches.Patch(color=c, label=f"{n} camp") for n, c in CAMP_COLORS.items()]
    ax.legend(handles=camp_patches, loc="upper left", fontsize=9, framealpha=0.9)
    panel_label(ax, "(a)")

    # (b) world map — bottom panel (full width)
    axm = fig.add_axes([0.045, 0.045, 0.92, 0.52], projection=ccrs.PlateCarree())
    axm.set_extent([-180, 180, -54, 71], crs=ccrs.PlateCarree())
    axm.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '110m',
                   facecolor=OCEAN_COLOR), zorder=0)
    axm.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m',
                   facecolor=LAND_COLOR), zorder=0)
    axm.add_feature(cfeature.NaturalEarthFeature('physical', 'coastline', '110m',
                   facecolor='none', edgecolor=COAST_COLOR, linewidth=0.4), zorder=1)
    axm.add_feature(cfeature.NaturalEarthFeature(
        'cultural', 'admin_0_boundary_lines_land', '110m',
        facecolor='none', edgecolor="#b0a89a", linewidth=0.35,
        linestyle=(0, (3, 2))), zorder=1)
    gl = axm.gridlines(draw_labels=False, linestyle=":", linewidth=0.45,
                       color="#8fa3b8", alpha=0.45)
    gl.set_zorder(1)
    for camp, color in CAMP_COLORS.items():
        sub = pg[pg["camp"] == camp]
        if len(sub) == 0:
            continue
        sizes = 6 + 90 * (sub["HCS_pct"] / sub["HCS_pct"].max())
        axm.scatter(sub["port_lon"], sub["port_lat"], s=sizes, c=color,
                    alpha=0.75, edgecolors="white", linewidths=0.3, zorder=6,
                    transform=ccrs.PlateCarree(), label=f"{camp} ({len(sub)})")
    sel = pg[pg["is_selected"]]
    axm.scatter(sel["port_lon"], sel["port_lat"], s=28, facecolors="none",
                edgecolors="black", linewidths=0.8, zorder=8,
                transform=ccrs.PlateCarree(), label="Selected (MILP)")
    axm.legend(loc="lower left", fontsize=9, framealpha=0.9)
    panel_label(axm, "(b)")

    plt.savefig(REFINE_FIG_DIR / f"fig_G2_control_{VESSEL_TYPE}.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: fig_G2_control_{VESSEL_TYPE}.png")


# ============================================================
# Fig E: Real efficiency-equity frontier (05d MOO)
# ============================================================

def fig_E():
    logger.info("[Fig E] Real MOO Frontier")
    moo_file = POLICY_DIR / f"moo_pareto_{VESSEL_TYPE}_bri_share.csv"
    if not moo_file.exists():
        logger.warning("  moo_pareto_bri_share.csv missing — run 05d first")
        return
    moo = pd.read_csv(moo_file)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    m = moo.sort_values("bri_share_pct")
    ax1.plot(m["bri_share_pct"], m["eff_obj_Mt"], "o-", color="#1565C0",
             linewidth=2, markersize=8, zorder=5)
    for _, r in m.iterrows():
        label = "free" if r["label"] == "share=free" else f"{r['bri_share_pct']:.0f}%"
        ax1.annotate(label, xy=(r["bri_share_pct"], r["eff_obj_Mt"]),
                     xytext=(4, 4), textcoords="offset points", fontsize=8)
    refs = [
        (78.9, 100.48, "M$^G$", "#2e7d32"),
        (83.3, 22.17, "M$^{EU}$", "#4169E1"),
        (100.0, 42.64, "M$^{BRI}$", "#DAA520"),
    ]
    for x, y, lab, col in refs:
        ax1.scatter(x, y, s=70, marker="D", c=col, edgecolors="black",
                    linewidths=0.6, zorder=6)
        ax1.annotate(lab, xy=(x, y), xytext=(6, -14), textcoords="offset points",
                     fontsize=8, color=col, fontweight="bold")
    free_eff = m.loc[m["label"] == "share=free", "eff_obj_Mt"].iloc[0]
    max_share = m["bri_share_pct"].max()
    top_eff = m.loc[m["bri_share_pct"] == max_share, "eff_obj_Mt"].iloc[0]
    premium = (free_eff - top_eff) / free_eff * 100
    ax1.annotate(f"Equity premium = {premium:.1f}%\n({free_eff:.1f} $\\to$ {top_eff:.1f} Mt)",
                 xy=(0.5, 0.1), xycoords="axes fraction", fontsize=9,
                 color="#D32F2F", fontweight="bold", ha="center",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", alpha=0.9))
    ax1.set_xlabel("BRI corridor abatement share (%)")
    ax1.set_ylabel("Global abatement (Mt CO$_2$e yr$^{-1}$)")
    ax1.grid(alpha=0.3)
    panel_label(ax1, "(a)")

    ax2.plot(m["bri_share_pct"], m["vac_obj_Mt"], "s-", color="#E74C3C",
             linewidth=2, markersize=7, zorder=5)
    ax2.axhline(m.loc[m["label"] == "share=free", "vac_obj_Mt"].iloc[0],
                color="#888888", linestyle=":", linewidth=1,
                label="Vacuum coverage at free optimum")
    for _, r in m.iterrows():
        ax2.annotate(f"{r['vac_obj_Mt']:.2f}", xy=(r["bri_share_pct"], r["vac_obj_Mt"]),
                     xytext=(0, 6), textcoords="offset points", fontsize=7,
                     ha="center")
    ax2.set_xlabel("BRI corridor abatement share (%)")
    ax2.set_ylabel("Governance-vacuum coverage (Mt CO$_2$e yr$^{-1}$)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    panel_label(ax2, "(b)")

    plt.tight_layout()
    save(f"fig_E_equity_{VESSEL_TYPE}")


# ============================================================
# Fig 1: Framework flowchart
# ============================================================

def fig_framework():
    logger.info("[Fig 1] Framework Flowchart")
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    W, H = 2.6, 1.15
    x0, y0 = 0.3, 8.3
    gap = 1.75

    def box(x, y, w, h, text, fc="#f4f7fb", ec="#1f4e79", fs=7.0, bold=False):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                           linewidth=1.1, edgecolor=ec, facecolor=fc)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal")

    def arr(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=11, linewidth=1.0,
                                     color="#333333"))

    box(x0, y0, W, H, "2024 AIS data\n7,471 containerships\n674,055 legs | 20,779 OD pairs",
        fs=6.8, bold=True)
    box(x0 + gap, y0, W, H, "Voyage-leg emission estimation\nIMO bottom-up WtW\n216.2 Mt CO$_2$e")
    box(x0 + 2 * gap, y0, W, H, "H3 attribution\n1,181,977 cells\n216.2 Mt CO$_2$e | Res5/Res7")
    box(x0 + 3 * gap, y0, W, H, "Corridor construction\n7,769 corridors (freq $\\geq$ 4)\n206.4 Mt (95.4%)")
    arr(x0 + W, y0 + H / 2, x0 + gap, y0 + H / 2)
    arr(x0 + gap + W, y0 + H / 2, x0 + 2 * gap, y0 + H / 2)
    arr(x0 + 2 * gap + W, y0 + H / 2, x0 + 3 * gap, y0 + H / 2)

    y1 = 6.2
    box(x0, y1, W, H, "Governance attribution classifier\nEU ETS endpoint rule | BRI 59 states\nFuelEU pressure P(h) $\\geq \\tau$",
        fc="#eef5ee", ec="#2e7d32", fs=6.8, bold=True)
    box(x0 + gap, y1, W, H, "Rule spheres (RQ1)\nBRI 58.0% | EU ETS 20.9%\nFuelEU 14.0% | Vacuum 7.1%")
    box(x0 + 2 * gap, y1, W, H, "Rule-Vacuum Index\nRVI = 6.3% strict\n151 high-emission lanes")
    arr(x0 + W / 2, y0, x0 + W / 2, y1 + H)
    arr(x0 + W + 0.3, y1 + H / 2, x0 + gap, y1 + H / 2)
    arr(x0 + gap + W, y1 + H / 2, x0 + 2 * gap, y1 + H / 2)

    y2 = 4.1
    box(x0, y2, W, H, "Rule-competition MILP family\nM$^G$ global | M$^{EU}$ rule sphere\nM$^{BRI}$ investment domain",
        fc="#eef5ee", ec="#2e7d32", fs=6.8, bold=True)
    box(x0 + gap, y2, W, H, "EU ETS revenue recycling\n$B + \\gamma P_{ETS} \\sum w_e E_e z_e$\nFuelEU top-100 forcing")
    box(x0 + 2 * gap, y2, W, H, "Competition utility matrix\nMPC: EU 77.9% | BRI 57.5%\nFuelEU 13.8--43.0%")
    arr(x0 + W / 2, y1, x0 + W / 2, y2 + H)
    arr(x0 + W + 0.3, y2 + H / 2, x0 + gap, y2 + H / 2)
    arr(x0 + gap + W, y2 + H / 2, x0 + 2 * gap, y2 + H / 2)

    y3 = 2.0
    box(x0, y3, W, H, "Multi-objective extension (MOO)\n$\\varepsilon$-constraint: efficiency $\\times$ vacuum\nBRI share $s \\in [0.80, 0.90]$",
        fc="#eef5ee", ec="#2e7d32", fs=6.8, bold=True)
    box(x0 + gap, y3, W, H, "Efficiency-equity frontier\nBRI 78.9% $\\to$ 90% costs 10.8%\n(equity premium 0.97%/pp)")
    box(x0 + 2 * gap, y3, W, H, "Hub control (RQ3)\nTop-20 ports 51.5%\nBRI 53.4% | EU 11.3%")
    arr(x0 + W / 2, y2, x0 + W / 2, y3 + H)
    arr(x0 + W + 0.3, y3 + H / 2, x0 + gap, y3 + H / 2)
    arr(x0 + gap + W, y3 + H / 2, x0 + 2 * gap, y3 + H / 2)

    y4 = 0.3
    box(x0, y4, 3 * gap + W, 0.85,
        "Policy outputs: governance geography (RQ1) | mismatch costs & utility matrix (RQ2) | "
        "hub control & carbon-price evidence (RQ3)",
        fc="#fdf6ec", ec="#b8860b", fs=7.0, bold=True)
    arr(x0 + 1.5 * gap, y3, x0 + 1.5 * gap, y4 + 0.85)

    save(f"fig1_framework_{VESSEL_TYPE}")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("REFINE GOV FIGURES (code version style)")
    logger.info("=" * 60)
    fig_G1()
    fig_G2()
    fig_E()
    fig_framework()
    logger.info("Done.")


