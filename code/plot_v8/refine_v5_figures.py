"""
refine_v5_figures.py — Layout fixes for 5 figures (v5)
=======================================================
Fixes (originals in plot_v4data/ are NOT modified):
  1. fig_imo_carbon  — reduce subplot spacing, remove suptitle
  2. fig1_framework  — redesign to eliminate text overlap (gap < W bug)
  3. fig10_bri       — move (b) subtitle inside axes
  4. fig8_multiperiod— reduce subplot spacing
  5. fig11_fueleu    — equalise top/bottom widths, premium colour scheme

Outputs to 04_图表_figures/v5_data/
"""

import sys
import json
import logging
from pathlib import Path
from importlib import import_module

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.gridspec import GridSpec

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import h3

# ─── Paths ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repository root
OUTPUT_DIR = PROJECT_ROOT / "data" / "derived"
REFINE_FIG_DIR = PROJECT_ROOT / "figures"
REFINE_FIG_DIR.mkdir(parents=True, exist_ok=True)

PLOT_V4_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PLOT_V4_DIR))
CODE_DIR = PROJECT_ROOT / "code" / "calc_v5"
sys.path.insert(0, str(CODE_DIR))
# this directory first so the v5 copies of refine_* modules win over the
# plot_v6final originals (which read v2/v4 data and would overwrite v6_final)
sys.path.insert(0, str(Path(__file__).parent))

from config import MULTIPERIOD_PARAMS

PROCESSED_DIR = OUTPUT_DIR / "processed_v5"
H3_DIR = OUTPUT_DIR / "h3_grid_v5"
RESULTS_DIR = OUTPUT_DIR / "milp_results_v5"
POLICY_DIR = OUTPUT_DIR / "policy_analysis_v5"
POLICY_V1 = OUTPUT_DIR / "policy_analysis_v5"
POLICY_V3 = OUTPUT_DIR / "policy_analysis_v5"

VESSEL_TYPE = "container"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("refine_v5")

# ─── Global Academic Style ──────────────────────────────────────
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
BORDER_COLOR = "#cccccc"


# ═══════════════════════════════════════════════════════════════
# Shared Utility Functions
# ═══════════════════════════════════════════════════════════════

def _premium_decor(ax):
    ax.add_feature(cfeature.NaturalEarthFeature(
        'cultural', 'admin_0_boundary_lines_land', '110m',
        facecolor='none', edgecolor=BORDER_COLOR, linewidth=0.35,
        linestyle=(0, (3, 2))), zorder=1)
    gl = ax.gridlines(draw_labels=False, linestyle=":", linewidth=0.45,
                      color="#8fa3b8", alpha=0.45)
    gl.set_zorder(1)
    return ax


def setup_basemap(fig, position=111, extent=None):
    ax = fig.add_subplot(position, projection=ccrs.PlateCarree())
    if extent is None:
        extent = [-180, 180, -58, 75]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '110m',
                   facecolor=OCEAN_COLOR), zorder=0)
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m',
                   facecolor=LAND_COLOR), zorder=0)
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'coastline', '110m',
                   facecolor='none', edgecolor=COAST_COLOR, linewidth=0.4), zorder=1)
    return _premium_decor(ax)


def setup_map_ax_small(fig, nrows, ncols, idx, extent=None):
    if extent is None:
        extent = [-180, 180, -56, 72]
    ax = fig.add_subplot(nrows, ncols, idx, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_COLOR, zorder=0)
    ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, zorder=0)
    ax.add_feature(cfeature.COASTLINE, edgecolor=COAST_COLOR, linewidth=0.3, zorder=1)
    return _premium_decor(ax)


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


# ═══════════════════════════════════════════════════════════════
# Fix 1: fig1_framework — REDESIGN (eliminate text overlap)
# Root cause: gap(1.75) < W(2.6) → boxes overlap horizontally
# Fix: gap > W, larger figure, adjusted proportions
# ═══════════════════════════════════════════════════════════════

def fig1_framework():
    logger.info("[Fig 1] Framework Flowchart (v5 redesign)")
    fig, ax = plt.subplots(figsize=(14, 9.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    W = 2.9
    H = 1.25
    x0 = 0.3
    gap = 3.3  # gap > W: no overlap, 0.4 spacing between boxes

    def box(x, y, w, h, text, fc="#f4f7fb", ec="#1f4e79", fs=7.0, bold=False):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                           linewidth=1.1, edgecolor=ec, facecolor=fc)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal",
                linespacing=1.35)

    def arr(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=11, linewidth=1.0,
                                     color="#333333"))

    # ── Stage 1: Data Foundation (4 boxes, blue) ──
    y0 = 8.3
    box(x0, y0, W, H,
        "2024 AIS data\n7,471 containerships\n674,055 legs | 20,779 OD pairs",
        fs=6.8, bold=True)
    box(x0 + gap, y0, W, H,
        "Voyage-leg emission estimation\nIMO bottom-up WtW\n216.2 Mt CO$_2$e")
    box(x0 + 2 * gap, y0, W, H,
        "H3 attribution\n1,181,977 cells\n216.2 Mt CO$_2$e | Res5/Res7")
    box(x0 + 3 * gap, y0, W, H,
        "Corridor construction\n7,769 corridors (freq $\\geq$ 4)\n206.4 Mt (95.4%)")
    arr(x0 + W, y0 + H / 2, x0 + gap, y0 + H / 2)
    arr(x0 + gap + W, y0 + H / 2, x0 + 2 * gap, y0 + H / 2)
    arr(x0 + 2 * gap + W, y0 + H / 2, x0 + 3 * gap, y0 + H / 2)

    # ── Stage 2: Governance Attribution (3 boxes, green) ──
    y1 = 6.3
    box(x0, y1, W, H,
        "Governance attribution classifier\nEU ETS endpoint rule | BRI 59 states\n"
        "FuelEU pressure P(h) $\\geq \\tau$",
        fc="#eef5ee", ec="#2e7d32", fs=6.8, bold=True)
    box(x0 + gap, y1, W, H,
        "Rule spheres (RQ1)\nBRI 58.0% | EU ETS 20.9%\nFuelEU 14.0% | Vacuum 7.1%")
    box(x0 + 2 * gap, y1, W, H,
        "Rule-Vacuum Index\nRVI = 6.3% strict\n151 high-emission lanes")
    arr(x0 + W / 2, y0, x0 + W / 2, y1 + H)
    arr(x0 + W + 0.3, y1 + H / 2, x0 + gap, y1 + H / 2)
    arr(x0 + gap + W, y1 + H / 2, x0 + 2 * gap, y1 + H / 2)

    # ── Stage 3: Rule-Competition MILP (3 boxes, green) ──
    y2 = 4.3
    box(x0, y2, W, H,
        "Rule-competition MILP family\nM$^G$ global | M$^{EU}$ rule sphere\n"
        "M$^{BRI}$ investment domain",
        fc="#eef5ee", ec="#2e7d32", fs=6.8, bold=True)
    box(x0 + gap, y2, W, H,
        "EU ETS revenue recycling\n$B + \\gamma P_{ETS} \\sum w_e E_e z_e$\n"
        "FuelEU top-100 forcing")
    box(x0 + 2 * gap, y2, W, H,
        "Competition utility matrix\nMPC: EU 77.9% | BRI 57.5%\nFuelEU 13.8--43.0%")
    arr(x0 + W / 2, y1, x0 + W / 2, y2 + H)
    arr(x0 + W + 0.3, y2 + H / 2, x0 + gap, y2 + H / 2)
    arr(x0 + gap + W, y2 + H / 2, x0 + 2 * gap, y2 + H / 2)

    # ── Stage 4: Multi-Objective Extension (3 boxes, green) ──
    y3 = 2.3
    box(x0, y3, W, H,
        "Multi-objective extension (MOO)\n$\\varepsilon$-constraint: efficiency $\\times$ vacuum\n"
        "BRI share $s \\in [0.80, 0.90]$",
        fc="#eef5ee", ec="#2e7d32", fs=6.8, bold=True)
    box(x0 + gap, y3, W, H,
        "Efficiency-equity frontier\nBRI 78.9% $\\to$ 90% costs 10.8%\n"
        "(equity premium 0.97%/pp)")
    box(x0 + 2 * gap, y3, W, H,
        "Hub control (RQ3)\nTop-20 ports 51.5%\nBRI 53.4% | EU 11.3%")
    arr(x0 + W / 2, y2, x0 + W / 2, y3 + H)
    arr(x0 + W + 0.3, y3 + H / 2, x0 + gap, y3 + H / 2)
    arr(x0 + gap + W, y3 + H / 2, x0 + 2 * gap, y3 + H / 2)

    # ── Bottom: Policy outputs (gold, full width) ──
    y4 = 0.4
    box(x0, y4, 3 * gap + W, 0.85,
        "Policy outputs: governance geography (RQ1) | mismatch costs & utility matrix (RQ2) | "
        "hub control & carbon-price evidence (RQ3)",
        fc="#fdf6ec", ec="#b8860b", fs=7.0, bold=True)
    arr(x0 + 1.5 * gap, y3, x0 + 1.5 * gap, y4 + 0.85)

    plt.savefig(REFINE_FIG_DIR / f"fig1_framework_{VESSEL_TYPE}.png", dpi=300,
                bbox_inches="tight", pad_inches=0.15)
    plt.close()
    logger.info("  Saved fig1_framework.")


# ═══════════════════════════════════════════════════════════════
# Fix 2: fig8_multiperiod — reduce subplot spacing
# Original: hspace=0.25 (too large), figure 14x9
# Fix: hspace=0.04, figure 14x8.2, tighter margins
# ═══════════════════════════════════════════════════════════════

def fig8_multiperiod():
    logger.info("[Fig 8] Multi-Period Frontier (v5 tight spacing)")
    periods = MULTIPERIOD_PARAMS["periods"]
    greens = ["#c8e6c9", "#81c784", "#4caf50", "#388e3c", "#2e7d32", "#1b5e20"]
    sub_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    fig = plt.figure(figsize=(14, 8.2))

    for idx, t in enumerate(periods):
        ax = setup_map_ax_small(fig, 3, 2, idx + 1)
        ax.text(0.02, 0.95, f"{sub_labels[idx]} {t}", transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))

        port_file = RESULTS_DIR / f"multiperiod_ports_{t}_{VESSEL_TYPE}_learn.csv"
        if not port_file.exists():
            port_file = RESULTS_DIR / f"multiperiod_ports_{t}_{VESSEL_TYPE}.csv"
        if port_file.exists():
            period_ports = pd.read_csv(port_file).dropna(subset=["port_lon", "port_lat"])
            marker_size = 30 + idx * 8
            ax.scatter(period_ports["port_lon"], period_ports["port_lat"],
                       c=greens[idx], s=marker_size, alpha=0.4,
                       edgecolors="none", zorder=3, transform=ccrs.PlateCarree())
            ax.scatter(period_ports["port_lon"], period_ports["port_lat"],
                       c="#1a1a1a", s=3, zorder=10, edgecolors="white",
                       linewidths=0.2, transform=ccrs.PlateCarree())
            n_ports = len(period_ports)
            ax.text(0.02, 0.05, f"n = {n_ports} ports", transform=ax.transAxes,
                    fontsize=11, color="#333333", fontweight="bold", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    legend_elements = [
        mpatches.Patch(facecolor=greens[0], alpha=0.4, edgecolor="none",
                       label="2025 (early stage)"),
        mpatches.Patch(facecolor=greens[2], alpha=0.4, edgecolor="none",
                       label="2035 (mid stage)"),
        mpatches.Patch(facecolor=greens[5], alpha=0.4, edgecolor="none",
                       label="2050 (full deployment)"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1a1a1a',
               markersize=4, label="Selected bunkering port"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=11,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.005))
    plt.subplots_adjust(top=0.98, bottom=0.075, left=0.01, right=0.99,
                        hspace=0.04, wspace=0.03)
    plt.savefig(REFINE_FIG_DIR / f"fig8_multiperiod_{VESSEL_TYPE}.png", dpi=300,
                bbox_inches="tight", pad_inches=0.1)
    plt.close()
    logger.info("  Saved fig8_multiperiod.")


# ═══════════════════════════════════════════════════════════════
# Fix 3: fig10_bri — move (b) subtitle inside axes
# Original: ax2.text(0.0, 1.07, ...) is OUTSIDE axes (y>1.0)
# Fix: move to (0.02, 0.97) with va="top" → inside axes
# ═══════════════════════════════════════════════════════════════

def fig10_bri():
    logger.info("[Fig 10] BRI Maritime Silk Road (v5 subtitle inside)")
    bri_corr_file = POLICY_DIR / f"bri_corridors_{VESSEL_TYPE}.parquet"
    summary_file = POLICY_DIR / f"bri_summary_{VESSEL_TYPE}.json"
    paths_file = H3_DIR / f"corridor_h3_paths_{VESSEL_TYPE}.parquet"

    if not bri_corr_file.exists() or not summary_file.exists():
        logger.warning("  BRI files not found — skipping")
        return

    with open(summary_file) as f:
        summary = json.load(f)

    corr_all = pd.read_parquet(PROCESSED_DIR / f"corridors_{VESSEL_TYPE}.parquet")
    act_file = RESULTS_DIR / f"activated_corridors_{VESSEL_TYPE}.csv"
    act_ids = set(pd.read_csv(act_file)["corridor_id"]) if act_file.exists() else set()
    bri_corr = pd.read_parquet(bri_corr_file)
    bri_ids = set(bri_corr.loc[bri_corr["is_bri"], "corridor_id"])
    n_act_bri = len(act_ids & bri_ids)
    n_act_nonbri = len(act_ids - bri_ids)
    n_nonbri_corr = max(len(corr_all) - len(bri_ids), 1)
    summary["bri_activation_rate_pct"] = n_act_bri / max(len(bri_ids), 1) * 100
    summary["nonbri_activation_rate_pct"] = n_act_nonbri / n_nonbri_corr * 100
    summary["activation_gap_pct"] = (summary["bri_activation_rate_pct"] -
                                     summary["nonbri_activation_rate_pct"])

    ports = pd.read_parquet(PROCESSED_DIR / "ports_global.parquet")
    ports_dedup = ports.drop_duplicates("port_code")
    m08 = import_module("08_bri_analysis")
    bri_countries = m08.BRI_MARITIME_COUNTRIES
    ports_dedup = ports_dedup.copy()
    ports_dedup["is_bri"] = ports_dedup["country_code"].isin(bri_countries)

    sel_file = RESULTS_DIR / f"selected_ports_{VESSEL_TYPE}.csv"
    selected_ports = set()
    if sel_file.exists():
        selected_ports = set(pd.read_csv(sel_file)["port_code"])
    ports_dedup["is_selected"] = ports_dedup["port_code"].isin(selected_ports)
    n_bri = int(ports_dedup["is_bri"].sum())
    n_sel_bri = int((ports_dedup["is_bri"] & ports_dedup["is_selected"]).sum())
    n_sel_nonbri = int((~ports_dedup["is_bri"] & ports_dedup["is_selected"]).sum())
    n_nonbri = len(ports_dedup) - n_bri
    summary["bri_selection_rate_pct"] = n_sel_bri / max(n_bri, 1) * 100
    summary["nonbri_selection_rate_pct"] = n_sel_nonbri / max(n_nonbri, 1) * 100
    summary["selection_gap_pct"] = (summary["bri_selection_rate_pct"] -
                                    summary["nonbri_selection_rate_pct"])

    fig = plt.figure(figsize=(13.5, 10.6))

    # ── (a) Top: Map ──
    ax = setup_basemap(fig, 211, extent=[-180, 180, -56, 72])
    ax.text(0.02, 0.97, "(a)  BRI Corridor & Port Coverage", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    if paths_file.exists():
        h3_paths = pd.read_parquet(paths_file)
        corridors_all = pd.read_parquet(PROCESSED_DIR / f"corridors_{VESSEL_TYPE}.parquet")
        bri_top = corridors_all[corridors_all["corridor_id"].isin(bri_ids)].nlargest(30, "E_e")
        nonbri_top = corridors_all[~corridors_all["corridor_id"].isin(bri_ids)].nlargest(20, "E_e")

        n_bri = len(bri_top)
        for rank, cid in enumerate(bri_top["corridor_id"]):
            cp = h3_paths[h3_paths["corridor_id"] == cid].sort_values("seq_order")
            if len(cp) > 1:
                frac = 1.0 - rank / max(n_bri - 1, 1)
                col = plt.cm.YlOrBr(0.35 + 0.55 * frac)
                lats = cp["h3_id"].apply(lambda h: h3.cell_to_latlng(h)[0]).values
                lons = cp["h3_id"].apply(lambda h: h3.cell_to_latlng(h)[1]).values
                for seg_lons, seg_lats in split_at_antimeridian(lons, lats):
                    ax.plot(seg_lons, seg_lats, color=col, alpha=0.85,
                            linewidth=1.8, zorder=5, transform=ccrs.Geodetic())

        for cid in nonbri_top["corridor_id"]:
            cp = h3_paths[h3_paths["corridor_id"] == cid].sort_values("seq_order")
            if len(cp) > 1:
                lats = cp["h3_id"].apply(lambda h: h3.cell_to_latlng(h)[0]).values
                lons = cp["h3_id"].apply(lambda h: h3.cell_to_latlng(h)[1]).values
                for seg_lons, seg_lats in split_at_antimeridian(lons, lats):
                    ax.plot(seg_lons, seg_lats, color="#9AA5B1", alpha=0.5,
                            linewidth=1.2, zorder=4, transform=ccrs.Geodetic())

    bri_sel = ports_dedup[ports_dedup["is_bri"] & ports_dedup["is_selected"]]
    bri_nosel = ports_dedup[ports_dedup["is_bri"] & ~ports_dedup["is_selected"]]
    nonbri_sel = ports_dedup[~ports_dedup["is_bri"] & ports_dedup["is_selected"]]

    ax.scatter(bri_sel["port_lon"], bri_sel["port_lat"], c="#FF4500", s=20,
               edgecolors="black", linewidths=0.4, zorder=10,
               transform=ccrs.PlateCarree(), label=f"BRI selected ({len(bri_sel)})")
    ax.scatter(bri_nosel["port_lon"], bri_nosel["port_lat"], c="#FF4500", s=8,
               marker="x", alpha=0.5, zorder=9,
               transform=ccrs.PlateCarree(), label=f"BRI not selected ({len(bri_nosel)})")
    ax.scatter(nonbri_sel["port_lon"], nonbri_sel["port_lat"], c="#4169E1", s=14,
               edgecolors="black", linewidths=0.3, zorder=10,
               transform=ccrs.PlateCarree(), label=f"Non-BRI selected ({len(nonbri_sel)})")
    corridor_handles = [
        Line2D([0], [0], color="#B8860B", linewidth=2.0,
               label="BRI corridors (gold by emission)"),
        Line2D([0], [0], color="#9AA5B1", linewidth=1.5, label="Non-BRI corridors"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=corridor_handles + handles, loc="lower left", fontsize=9.5, framealpha=0.9)

    # ── (b) Bottom: Bar chart ──
    ax2 = fig.add_subplot(212)
    # FIX: subtitle moved INSIDE axes (was y=1.07 outside; now y=0.95 va="top"
    # with no bbox so nothing extends above the axes spine)
    ax2.text(0.015, 0.95, "(b)  BRI vs Non-BRI: Emission Share, Activation & Selection",
             transform=ax2.transAxes, fontsize=12, fontweight="bold",
             va="top", ha="left")

    categories = ["Emission\nShare (%)", "Corridor\nActivation (%)", "Port\nSelection (%)"]
    bri_vals = [summary["bri_emission_share_pct"],
                summary["bri_activation_rate_pct"],
                summary["bri_selection_rate_pct"]]
    nonbri_vals = [100 - summary["bri_emission_share_pct"],
                   summary["nonbri_activation_rate_pct"],
                   summary["nonbri_selection_rate_pct"]]

    x = np.arange(len(categories))
    width = 0.35
    bars1 = ax2.bar(x - width/2, bri_vals, width, label="BRI", color="#DAA520",
                    edgecolor="black", linewidth=0.5)
    bars2 = ax2.bar(x + width/2, nonbri_vals, width, label="Non-BRI", color="#4169E1",
                    edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Percentage (%)", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=11)
    ax2.tick_params(axis="y", labelsize=10)
    ax2.legend(fontsize=11.5)
    ax2.set_ylim(0, 108)
    ax2.grid(axis="y", alpha=0.3)

    for bar in bars1:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")

    gap_text = (f"Activation gap: {summary['activation_gap_pct']:+.1f} pp\n"
                f"Selection gap: {summary['selection_gap_pct']:+.1f} pp")
    ax2.text(0.98, 0.92, gap_text, transform=ax2.transAxes, fontsize=10,
             va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout(pad=0.8)
    plt.savefig(REFINE_FIG_DIR / f"fig10_bri_{VESSEL_TYPE}.png", dpi=300,
                bbox_inches="tight", pad_inches=0.1)
    plt.close()
    logger.info("  Saved fig10_bri.")


# ═══════════════════════════════════════════════════════════════
# Fix 4: fig_imo_carbon — reduce subplot spacing, remove suptitle
# Original: hspace=0.18, wspace=0.08, suptitle present
# Fix: hspace=0.05, wspace=0.02, no suptitle, tighter figure
# ═══════════════════════════════════════════════════════════════

def fig_imo_carbon():
    logger.info("[Fig IMO Carbon] (v5 tight, no suptitle)")

    mab = import_module("refine_v2_abatement")

    corridors = pd.read_parquet(PROCESSED_DIR / "corridors_container.parquet")
    paths = pd.read_parquet(H3_DIR / "corridor_h3_paths_container.parquet")
    summary = pd.read_csv(POLICY_DIR / "imo_carbon_summary_container.csv")
    summary["abate_mt"] = summary["abatement_tco2e"] / 1e6

    ref_act = pd.read_csv(RESULTS_DIR / "activated_corridors_container.csv")
    ref_ids = set(ref_act["corridor_id"])

    CARBON_PRICES = [380, 1000, 2000, 5000]
    LABELS = {380: "IMO cap 380", 1000: "1,000", 2000: "2,000", 5000: "5,000"}

    scen_ids = {}
    for c in CARBON_PRICES:
        f = POLICY_DIR / f"imo_carbon_activated_{c}_container.csv"
        scen_ids[c] = set(pd.read_csv(f)["corridor_id"]) if f.exists() else set()

    def setup_map_ax_gs(fig, gs_slot):
        ax = fig.add_subplot(gs_slot, projection=ccrs.PlateCarree())
        ax.set_extent([-180, 180, -52, 71], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '110m',
                       facecolor=OCEAN_COLOR), zorder=0)
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m',
                       facecolor=LAND_COLOR), zorder=0)
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 'coastline', '110m',
                       facecolor='none', edgecolor=COAST_COLOR, linewidth=0.4), zorder=1)
        ax.add_feature(cfeature.NaturalEarthFeature(
            'cultural', 'admin_0_boundary_lines_land', '110m',
            facecolor='none', edgecolor="#b0a89a", linewidth=0.3,
            linestyle=(0, (3, 2))), zorder=1)
        gl = ax.gridlines(draw_labels=False, linestyle=":", linewidth=0.4,
                          color="#8fa3b8", alpha=0.4)
        gl.set_zorder(1)
        return ax

    def abate_map_raster(corr_ids, scale_to=None):
        act = corridors[corridors["corridor_id"].isin(corr_ids)].copy()
        act = act[["corridor_id", "E_e", "eta_green_methanol", "alpha_e"]]
        act["abatement"] = act["E_e"] * act["eta_green_methanol"].fillna(0) \
            * act["alpha_e"].fillna(0)
        ab_cells = mab.abatement_to_cells(act, paths, scale_to=scale_to)
        if len(ab_cells) == 0:
            return None, None, None
        ab_cells = mab.cell_df_with_lonlat(ab_cells)
        return mab.kde_rasterize(ab_cells, sigma=0.5)

    fig = plt.figure(figsize=(12.2, 7.6))
    # FIX: much tighter spacing, no suptitle
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.05, 1, 1],
                  hspace=0.05, wspace=0.02)

    # (a) abatement comparison bar
    ax_a = fig.add_subplot(gs[0, 0])
    cats = ["M$^G$", "380", "1,000", "2,000", "5,000"]
    vals = [summary.iloc[0]["abate_mt"]] + \
           [summary.loc[summary["carbon_price"] == str(c), "abate_mt"].iloc[0]
            for c in CARBON_PRICES]
    colors = ["#1a9850"] + ["#d9d9d9", "#a6d96a", "#66bd63", "#1a9850"]
    bars = ax_a.bar(cats, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax_a.set_ylabel("Abatement (Mt CO$_2$e yr$^{-1}$)")
    ax_a.set_ylim(0, max(vals) * 1.25)
    for b, v in zip(bars, vals):
        ax_a.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}",
                  ha="center", fontsize=9, fontweight="bold")
    ax_a.axhline(vals[0], color="#1a9850", linestyle="--", linewidth=1, alpha=0.6)
    ax_a.text(0.02, 0.97, "(a)  Market vs Planning Abatement",
              transform=ax_a.transAxes, fontsize=11, fontweight="bold", va="top")
    ax_a.grid(axis="y", alpha=0.3)
    ax_a.tick_params(axis="x", labelsize=9)

    # (b) M^G reference density
    ax_b = setup_map_ax_gs(fig, gs[1, 0])
    r, LON, LAT = abate_map_raster(ref_ids)
    if r is not None:
        pv = r.compressed()
        norm = matplotlib.colors.LogNorm(vmin=max(np.percentile(pv, 10), 0.1),
                                         vmax=np.percentile(pv, 99))
        ax_b.pcolormesh(LON, LAT, r.T, cmap="BuGn", norm=norm, alpha=0.9,
                        shading="auto", transform=ccrs.PlateCarree(), zorder=2)
    ax_b.text(0.02, 0.95, f"(b)  M$^G$ planning   {vals[0]:.1f} Mt",
              transform=ax_b.transAxes, fontsize=10.5, fontweight="bold", va="top",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    # (c)-(f) carbon price scenarios
    vmax_all = 0.0
    rasters = {}
    for c in CARBON_PRICES:
        r, _, _ = abate_map_raster(scen_ids[c])
        rasters[c] = r
        if r is not None and r.count() > 0:
            vmax_all = max(vmax_all, float(r.max()))

    slots = [gs[2, 0], gs[0, 1], gs[1, 1], gs[2, 1]]
    vmin_shared = None
    for i, c in enumerate(CARBON_PRICES):
        ax = setup_map_ax_gs(fig, slots[i])
        r = rasters[c]
        ab_mt = summary.loc[summary["carbon_price"] == str(c), "abate_mt"].iloc[0]
        if r is None or r.count() == 0:
            ax.text(0.5, 0.5, "no economically\nviable corridor",
                    transform=ax.transAxes, fontsize=11, ha="center", va="center",
                    fontweight="bold", color="#666666")
        else:
            pv = r.compressed()
            vmin_shared = vmin_shared or max(np.percentile(pv, 10), 0.1)
            norm = matplotlib.colors.LogNorm(vmin=vmin_shared, vmax=vmax_all)
            ax.pcolormesh(LON, LAT, r.T, cmap="BuGn", norm=norm, alpha=0.9,
                          shading="auto", transform=ccrs.PlateCarree(), zorder=2)
        ax.text(0.02, 0.94, f"({chr(99 + i)})  {LABELS[c]} USD/t   {ab_mt:.1f} Mt",
                transform=ax.transAxes, fontsize=10.5, fontweight="bold", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    # FIX: no suptitle (removed)
    plt.savefig(REFINE_FIG_DIR / "fig_imo_carbon_container.png", dpi=300,
                bbox_inches="tight", pad_inches=0.08)
    plt.close()
    logger.info("  Saved fig_imo_carbon.")


# ═══════════════════════════════════════════════════════════════
# Fix 5: fig11_fueleu — equalise top/bottom widths + premium palette
# Root cause: top subplot's colorbar steals width via tight_layout,
#   making ax (top) narrower than ax2 (bottom).
# Fix: explicit add_axes with identical [left, width]; colorbar on
#   a separate thin axes to the right.  Custom "indigo→teal→amber"
#   colormap replaces viridis for a more sophisticated look; alpha
#   lowered to 0.72 so the ocean base shows through naturally.
# ═══════════════════════════════════════════════════════════════

def fig11_fueleu():
    logger.info("[Fig 11] FuelEU Maritime (v5 equal-width + premium palette)")

    pressure_file = POLICY_DIR / f"fueleu_pressure_{VESSEL_TYPE}.parquet"
    comparison_file = POLICY_DIR / f"fueleu_comparison_{VESSEL_TYPE}.json"
    changes_file = POLICY_DIR / f"fueleu_port_changes_{VESSEL_TYPE}.csv"

    if not pressure_file.exists():
        logger.warning("  FuelEU pressure file not found — skipping")
        return

    pressure_df = pd.read_parquet(pressure_file)
    pressure_df = pressure_df[pressure_df["lat"].abs() < 72]
    with open(comparison_file) as f:
        comparison = json.load(f)

    # ── Premium custom colormap: deep indigo → teal → amber ──
    from matplotlib.colors import LinearSegmentedColormap
    premium_cmap = LinearSegmentedColormap.from_list(
        "fueleu_premium",
        [(0.00, "#1a1a3e"),   # deep indigo (low pressure)
         (0.20, "#2c3e72"),   # navy
         (0.40, "#1a6e8e"),   # teal
         (0.60, "#3dae8a"),   # soft green-teal
         (0.78, "#d4a843"),   # muted gold
         (1.00, "#c25e2a")],  # warm amber (high pressure)
        N=256)

    fig = plt.figure(figsize=(14, 10.2))

    # ── Explicit axes positioning: both maps share identical width ──
    map_left = 0.008
    map_width = 0.88          # both maps use this exact width
    cax_left = 0.895          # colorbar strip
    cax_width = 0.018
    top_h = 0.42
    bot_h = 0.42
    top_bottom = 0.54
    bot_bottom = 0.06

    # (a) Top map
    ax = fig.add_axes([map_left, top_bottom, map_width, top_h],
                      projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -56, 72], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '110m',
                   facecolor=OCEAN_COLOR), zorder=0)
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m',
                   facecolor=LAND_COLOR), zorder=0)
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'coastline', '110m',
                   facecolor='none', edgecolor=COAST_COLOR, linewidth=0.4), zorder=1)
    _premium_decor(ax)

    ax.text(0.015, 0.96, "(a)  FuelEU Compliance Pressure Field",
            transform=ax.transAxes, fontsize=12.5, fontweight="bold",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    # Rasterise pressure onto 0.5° grid
    lon_bins = np.arange(-180, 180.5, 0.5)
    lat_bins = np.arange(-72, 72.5, 0.5)
    lon_idx = np.clip(np.digitize(pressure_df["lon"], lon_bins) - 1, 0, len(lon_bins) - 2)
    lat_idx = np.clip(np.digitize(pressure_df["lat"], lat_bins) - 1, 0, len(lat_bins) - 2)
    raster = np.zeros((len(lon_bins) - 1, len(lat_bins) - 1))
    np.maximum.at(raster, (lon_idx, lat_idx), pressure_df["fueleu_pressure"].values)
    raster = np.ma.masked_where(raster == 0, raster)
    lon_c = (lon_bins[:-1] + lon_bins[1:]) / 2
    lat_c = (lat_bins[:-1] + lat_bins[1:]) / 2
    LON, LAT = np.meshgrid(lon_c, lat_c)

    # Premium look: custom cmap + refined alpha
    mesh = ax.pcolormesh(LON, LAT, raster.T, cmap=premium_cmap, alpha=0.72,
                         shading="auto", transform=ccrs.PlateCarree(), zorder=2)

    # Colorbar on a dedicated axes (does NOT steal map width)
    cax = fig.add_axes([cax_left, top_bottom, cax_width, top_h])
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("Compliance Pressure P(h)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # Exposure threshold contour
    try:
        ax.contour(LON, LAT, raster.T, levels=[0.30], colors="white",
                   linestyles="--", linewidths=1.4,
                   transform=ccrs.PlateCarree(), zorder=3)
        ax.plot([], [], color="white", linestyle="--", linewidth=1.6,
                label="Exposure threshold $\\tau$ = 0.3")
        ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    except Exception:
        pass

    # (b) Bottom map — same width as top
    ax2 = fig.add_axes([map_left, bot_bottom, map_width, bot_h],
                       projection=ccrs.PlateCarree())
    ax2.set_extent([-180, 180, -56, 72], crs=ccrs.PlateCarree())
    ax2.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '110m',
                    facecolor=OCEAN_COLOR), zorder=0)
    ax2.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m',
                    facecolor=LAND_COLOR), zorder=0)
    ax2.add_feature(cfeature.NaturalEarthFeature('physical', 'coastline', '110m',
                    facecolor='none', edgecolor=COAST_COLOR, linewidth=0.4), zorder=1)
    _premium_decor(ax2)

    ax2.text(0.015, 0.96, f"(b)  FuelEU-Forced Siting Changes  "
                          f"(+{comparison.get('new_ports_added', 0)} / "
                          f"\u2212{comparison.get('ports_removed', 0)})"
                          f"   ·   top-{comparison.get('eu_ports_forced', 100)} "
                          f"EU ports forced, 20% budget",
             transform=ax2.transAxes, fontsize=12.5, fontweight="bold",
             va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    if changes_file.exists():
        changes = pd.read_csv(changes_file)
        added = changes[changes["change"] == "added"]
        removed = changes[changes["change"] == "removed"]
        if len(added) > 0:
            ax2.scatter(added["port_lon"], added["port_lat"], c="#2ecc71", s=180,
                        alpha=0.22, edgecolors="none", zorder=9,
                        transform=ccrs.PlateCarree())
            ax2.scatter(added["port_lon"], added["port_lat"], c="#2ecc71", s=58,
                        edgecolors="white", linewidths=0.8, zorder=10,
                        transform=ccrs.PlateCarree(), label=f"Added ({len(added)})")
        if len(removed) > 0:
            ax2.scatter(removed["port_lon"], removed["port_lat"], c="#e74c3c", s=180,
                        alpha=0.20, edgecolors="none", zorder=9,
                        transform=ccrs.PlateCarree())
            ax2.scatter(removed["port_lon"], removed["port_lat"], c="#e74c3c", s=62,
                        marker="X", edgecolors="white", linewidths=1.1, zorder=10,
                        transform=ccrs.PlateCarree(), label=f"Removed ({len(removed)})")
        ax2.legend(loc="lower left", fontsize=10.5, framealpha=0.9)

    obj_pct = comparison.get("obj_change_pct")
    if obj_pct is None:
        obj_pct = comparison.get("MPC_global_pct")
    obj_text = (f"Global abatement: \u2212{obj_pct:.1f}%" if isinstance(obj_pct, (int, float))
                else "Global abatement: n/a")
    ax2.text(0.985, 0.95, obj_text, transform=ax2.transAxes, fontsize=9,
             va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.savefig(REFINE_FIG_DIR / f"fig11_fueleu_{VESSEL_TYPE}.png", dpi=300,
                bbox_inches="tight", pad_inches=0.08)
    plt.close()
    logger.info("  Saved fig11_fueleu.")


# ═══════════════════════════════════════════════════════════════
# Fix 6: fig_E_equity — restructure left subplot
# Root cause: BRI share data points are 79.6-90%, but x-axis spans
#   0-100% (M^EU at 21.4% creates the illusion of full range).
#   The frontier curve sits in the right 15%, leaving 75% empty.
# Fix: zoom x-axis to 70-100%, add shaded zones, marginal bars,
#   and annotations to fill the visual space meaningfully.
# ═══════════════════════════════════════════════════════════════

def fig_E_equity():
    logger.info("[Fig E] Efficiency-Equity Frontier (v5 restructured)")
    moo_file = POLICY_DIR / f"moo_pareto_{VESSEL_TYPE}_bri_share.csv"
    if not moo_file.exists():
        logger.warning("  moo_pareto_bri_share.csv missing — skipping")
        return
    moo = pd.read_csv(moo_file)

    fig = plt.figure(figsize=(14, 5.8))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.15, 1], wspace=0.15)

    m = moo.sort_values("bri_share_pct")

    # ── (a) Left: restructured frontier — zoomed 70-100% ──
    ax1 = fig.add_subplot(gs[0, 0])

    # Shaded zones for visual richness
    ax1.axvspan(70, 78.9, alpha=0.06, color="#888888", zorder=0)
    ax1.axvspan(78.9, 90, alpha=0.08, color="#1565C0", zorder=0)
    ax1.axvspan(90, 101, alpha=0.06, color="#DAA520", zorder=0)
    ax1.text(74.8, 101.5, "Sub-optimal\nzone", ha="center", va="top",
             fontsize=7.5, color="#666666", style="italic")
    ax1.text(84.8, 101.5, "Pareto frontier", ha="center", va="top",
             fontsize=8, color="#1565C0", fontweight="bold")
    ax1.text(95.5, 101.5, "Equity\npremium\nzone", ha="center", va="top",
             fontsize=7.5, color="#8B6914", style="italic")

    # Fill under curve for premium look
    ax1.fill_between(m["bri_share_pct"], m["eff_obj_Mt"],
                     m["eff_obj_Mt"].min() - 1, alpha=0.12, color="#1565C0", zorder=2)
    ax1.plot(m["bri_share_pct"], m["eff_obj_Mt"], "o-", color="#1565C0",
             linewidth=1.8, markersize=7, markeredgecolor="white",
             markeredgewidth=0.6, zorder=5)
    for _, r in m.iterrows():
        label = "free" if r["label"] == "share=free" else f"{r['bri_share_pct']:.0f}%"
        ax1.annotate(label, xy=(r["bri_share_pct"], r["eff_obj_Mt"]),
                     xytext=(6, 5), textcoords="offset points", fontsize=8,
                     fontweight="bold", color="#333333")
    refs = [
        (78.9, 100.48, "M$^G$", "#2e7d32"),
        # M^EU (83.3% share, 22.2 Mt) and M^BRI (100% share, 42.6 Mt) lie far
        # below the zoomed y-range (88-102 Mt) and are intentionally not shown
    ]
    for x, y, lab, col in refs:
        if 70 <= x <= 101:
            ax1.scatter(x, y, s=60, marker="D", c=col, edgecolors="black",
                        linewidths=0.5, zorder=6)
            ax1.annotate(lab, xy=(x, y), xytext=(-2, -20), textcoords="offset points",
                         fontsize=8, color=col, fontweight="bold")

    free_eff = m.loc[m["label"] == "share=free", "eff_obj_Mt"].iloc[0]
    max_share = m["bri_share_pct"].max()
    top_eff = m.loc[m["bri_share_pct"] == max_share, "eff_obj_Mt"].iloc[0]
    premium = (free_eff - top_eff) / free_eff * 100
    ax1.annotate(f"Equity premium = {premium:.1f}%\n({free_eff:.1f} $\\to$ {top_eff:.1f} Mt)",
                 xy=(0.45, 0.12), xycoords="axes fraction", fontsize=9,
                 color="#D32F2F", fontweight="bold", ha="center",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", alpha=0.9))
    ax1.set_xlabel("BRI corridor abatement share (%)", fontsize=11)
    ax1.set_ylabel("Global abatement (Mt CO$_2$e yr$^{-1}$)", fontsize=11)
    ax1.set_xlim(70, 101)
    ax1.set_ylim(m["eff_obj_Mt"].min() - 1.5, m["eff_obj_Mt"].max() + 1.5)
    ax1.grid(alpha=0.25, linestyle=":")
    ax1.text(0.015, 0.97, "(a)", transform=ax1.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left")

    # ── (b) Right: vacuum coverage (refined style) ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.fill_between(m["bri_share_pct"], m["vac_obj_Mt"],
                     0, alpha=0.10, color="#E74C3C", zorder=2)
    ax2.plot(m["bri_share_pct"], m["vac_obj_Mt"], "s-", color="#E74C3C",
             linewidth=1.8, markersize=6, markeredgecolor="white",
             markeredgewidth=0.5, zorder=5)
    ax2.axhline(m.loc[m["label"] == "share=free", "vac_obj_Mt"].iloc[0],
                color="#888888", linestyle=(0, (4, 3)), linewidth=1,
                label="Vacuum at free optimum")
    for _, r in m.iterrows():
        ax2.annotate(f"{r['vac_obj_Mt']:.2f}", xy=(r["bri_share_pct"], r["vac_obj_Mt"]),
                     xytext=(8, 5), textcoords="offset points", fontsize=7.5,
                     ha="left", fontweight="bold", color="#333333")
    ax2.set_xlabel("BRI corridor abatement share (%)", fontsize=11)
    ax2.set_ylabel("Governance-vacuum coverage (Mt CO$_2$e yr$^{-1}$)", fontsize=11)
    ax2.set_xlim(70, 101)
    ax2.legend(fontsize=11, loc="upper right")
    ax2.grid(alpha=0.25, linestyle=":")
    ax2.text(0.015, 0.97, "(b)", transform=ax2.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left")

    plt.savefig(REFINE_FIG_DIR / f"fig_E_equity_{VESSEL_TYPE}.png", dpi=300,
                bbox_inches="tight", pad_inches=0.1)
    plt.close()
    logger.info("  Saved fig_E_equity.")


# ═══════════════════════════════════════════════════════════════
# Fix 7: fig_C_carbon_cascade — move legend up away from inset
# ═══════════════════════════════════════════════════════════════

def fig_C_carbon_cascade():
    logger.info("[Fig C] Carbon Cascade (v5 legend repositioned)")
    from scipy.ndimage import label as nd_label, binary_dilation

    bp_file = POLICY_DIR / f"breakeven_prices_{VESSEL_TYPE}.csv"
    gov_file = POLICY_DIR / f"governance_corridors_{VESSEL_TYPE}.parquet"
    if not bp_file.exists():
        logger.warning("  breakeven_prices missing — skipping")
        return

    bp = pd.read_csv(bp_file)
    bp = bp.dropna(subset=["p_star_usd"])
    if gov_file.exists():
        gov = pd.read_parquet(gov_file)[["corridor_id", "gov_type"]]
        bp = bp.merge(gov, on="corridor_id", how="left")
        bp["gov_type"] = bp["gov_type"].fillna("Vacuum")

    p = bp["p_star_usd"].clip(100, 1_000_000).values
    abate = bp["dE_e"].values
    prices = np.logspace(2, 6, 160)
    cum_corridors = np.array([(p <= px).sum() for px in prices])
    cum_abatement = np.array([abate[p <= px].sum() / 1e6 for px in prices])
    n_total = len(p)

    fig, ax_main = plt.subplots(figsize=(14.5, 8.3))
    ax_main.fill_between(prices, cum_abatement, alpha=0.15, color="#1565C0")
    ax_main.plot(prices, cum_abatement, color="#1565C0", linewidth=2.0,
                 label="Cumulative abatement (green-methanol scenario)")
    ax2 = ax_main.twinx()
    ax2.plot(prices, cum_corridors, color="#FF7043", linewidth=1.5, linestyle="--",
             label="Corridors activated")
    ax2.set_ylabel("Number of corridors activated", fontsize=11, color="#FF7043")
    ax2.tick_params(axis="y", labelcolor="#FF7043")

    anchors = [("EU ETS (~80)", 100, "#4CAF50"),
               ("IMO proposals (max 380)", 380, "#FFC107"),
               ("p$_{25}$ = 2.1k", 2_059, "#9C27B0"),
               ("median = 8.1k", 8_053, "#D32F2F")]
    ylim_top = np.nanmax(cum_abatement)
    for label_text, price, color in anchors:
        ax_main.axvline(price, color=color, linestyle="--", linewidth=1.0, alpha=0.6)
        ax_main.annotate(label_text, xy=(price, ylim_top * 0.97),
                         fontsize=10, color=color, ha="center", va="top",
                         bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                                   edgecolor=color, alpha=0.9, linewidth=0.5))
    ax_main.set_xscale("log")
    ax_main.set_xlabel("Carbon Price (USD / tCO$_{2}$)", fontsize=11)
    ax_main.set_ylabel("Cumulative Abatement (Mt CO$_2$e yr$^{-1}$)", fontsize=11)
    ax_main.set_xlim(200, 1_000_000)

    ax_inset = fig.add_axes([0.55, 0.18, 0.33, 0.30], facecolor="white")
    for sp in ax_inset.spines.values():
        sp.set_edgecolor("#555555")
        sp.set_linewidth(0.8)
    camp_colors = {"BRI_only": "#DAA520", "EU_ETS": "#4169E1",
                   "FuelEU_exposed": "#8E44AD", "Vacuum": "#E74C3C"}
    bins = np.logspace(2, 6, 40)
    for gt, col in camp_colors.items():
        sub = bp.loc[bp["gov_type"] == gt, "p_star_usd"].clip(100, 1_000_000)
        if len(sub) > 0:
            ax_inset.hist(sub, bins=bins, alpha=0.55, color=col,
                          label=f"{gt} (n={len(sub):,})", density=True)
    ax_inset.set_xscale("log")
    ax_inset.set_xlim(100, 1_000_000)
    ax_inset.set_xlabel("Break-even price (USD/tCO$_2$, log)", fontsize=9.5,
                        fontweight="bold")
    ax_inset.set_ylabel("Density", fontsize=9.5, fontweight="bold")
    ax_inset.set_title("Distribution of $p^*_e$ by governance camp", fontsize=11,
                       fontweight="bold")
    ax_inset.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax_inset.tick_params(labelsize=8)

    # FIX: legend moved to upper left with bbox, clear of the inset
    lines1, labels1 = ax_main.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_main.legend(lines1 + lines2, labels1 + labels2,
                   loc="upper left", bbox_to_anchor=(0.01, 0.78),
                   fontsize=9, framealpha=0.92, edgecolor="#cccccc",
                   fancybox=True, borderpad=0.8)
    ax_main.grid(alpha=0.15, which="both")

    plt.savefig(REFINE_FIG_DIR / f"fig_C_carbon_cascade_{VESSEL_TYPE}.png", dpi=300,
                bbox_inches="tight", pad_inches=0.1)
    plt.close()
    logger.info(f"  Saved fig_C_carbon_cascade (n={n_total} corridors).")


# ═══════════════════════════════════════════════════════════════
# Fix 8: fig3_pareto — shorter guide lines, no overlap, premium
#   line style, thinner main lines
# ═══════════════════════════════════════════════════════════════

def fig3_pareto():
    logger.info("[Fig 3] Pareto Frontier (v5 premium annotations)")
    fig, ax = plt.subplots(1, 1, figsize=(8.6, 6.2))
    # Premium palette: deep teal + muted indigo
    colors = {"green_ammonia": "#0D7377", "green_methanol": "#3A5BA0"}
    markers = {"green_ammonia": "D", "green_methanol": "o"}

    df_me = None
    for fuel in ["green_ammonia", "green_methanol"]:
        fpath = RESULTS_DIR / f"pareto_{fuel}_{VESSEL_TYPE}.csv"
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath)
        if "methanol" in fuel:
            df_me = df
        label = ("Green ammonia ($R_f$ = 5,000 nm)" if "ammonia" in fuel
                 else "Green methanol ($R_f$ = 8,000 nm)")
        # FIX: thinner lines (2.4 -> 1.6), premium colors
        ax.plot(df["budget_fraction"] * 100, df["objective_tco2e"] / 1e6,
                marker=markers[fuel], color=colors[fuel], linewidth=1.6,
                markersize=7, markeredgecolor="white", markeredgewidth=0.6,
                label=label, zorder=4)

    if df_me is not None:
        r20 = df_me[df_me["budget_fraction"] == 0.20].iloc[0]
        ax.axvline(20, color="#888888", linestyle=":", linewidth=1.0,
                   alpha=0.7, zorder=2)

        # FIX: much shorter guide line (was 12.0, now 16.5 — closer to 20)
        y20 = r20["objective_tco2e"] / 1e6
        ax.annotate("reference scenario\n(20% budget)",
                    xy=(20, y20),
                    xytext=(16.5, y20 + 1.0), fontsize=9.5,
                    ha="center",
                    arrowprops=dict(arrowstyle="->", color="#555555",
                                    lw=0.8, connectionstyle="arc3,rad=0.15"),
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#f8f4e6",
                              alpha=0.9, edgecolor="#bbb"))

        # FIX: shorter guide for diminishing returns (was 24.8, now 27.5)
        r30 = df_me[df_me["budget_fraction"] == 0.30].iloc[0]
        gain = (r30["objective_tco2e"] - r20["objective_tco2e"]) / 1e6
        y30 = r30["objective_tco2e"] / 1e6
        ax.annotate(f"+{gain:.1f} Mt only\n(diminishing returns)",
                    xy=(30, y30),
                    xytext=(27.5, y30 - 4.8), fontsize=9,
                    color="#555555", ha="center",
                    arrowprops=dict(arrowstyle="->", color="#888888",
                                    lw=0.7, connectionstyle="arc3,rad=-0.15"),
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#f0f0f0",
                              alpha=0.9, edgecolor="#ccc"))

    ax.set_xlabel("Budget Fraction (%)", fontsize=12)
    ax.set_ylabel("Annual CO$_2$e Abatement (Mt yr$^{-1}$)", fontsize=12)
    ax.legend(fontsize=10.5, loc="upper left", framealpha=0.9,
              edgecolor="#ccc", fancybox=True)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.set_xticks([5, 10, 20, 30])

    if df_me is not None:
        axi = ax.inset_axes([0.50, 0.09, 0.37, 0.30])
        axi.set_facecolor("white")
        x = df_me["budget_fraction"] * 100
        axi.bar(x, df_me["n_ports"], width=3.4, color="#0D7377",
                edgecolor="#07494C", linewidth=0.4, alpha=0.85)
        axi2 = axi.twinx()
        axi2.plot(x, df_me["n_corridors"], color="#3A5BA0", marker="o",
                  markersize=4, linewidth=1.3, linestyle="-")
        axi.set_title("Infrastructure scale (methanol)", fontsize=9,
                      fontweight="bold")
        axi.set_xlabel("Budget (%)", fontsize=8)
        axi.set_ylabel("Ports upgraded", fontsize=8, color="#0D7377")
        axi2.set_ylabel("Corridors activated", fontsize=8, color="#3A5BA0")
        axi.tick_params(labelsize=7.5, colors="#0D7377")
        axi2.tick_params(labelsize=7.5, colors="#3A5BA0")
        axi.set_xticks([5, 10, 20, 30])
        for sp in list(axi.spines.values()) + list(axi2.spines.values()):
            sp.set_linewidth(0.8)

    plt.tight_layout(pad=1.2)
    plt.savefig(REFINE_FIG_DIR / f"fig3_pareto_{VESSEL_TYPE}.png", dpi=300,
                bbox_inches="tight", pad_inches=0.1)
    plt.close()
    logger.info("  Saved fig3_pareto.")


# ═══════════════════════════════════════════════════════════════
# Fix 9: fig7_eu_ets_h3 — enrich map, refine port visuals
# ═══════════════════════════════════════════════════════════════

def fig7_eu_ets_h3():
    logger.info("[Fig 7] EU ETS Coverage (v5 enriched)")
    from scipy.ndimage import label as nd_label, binary_dilation

    eu_file = H3_DIR / f"h3_eu_ets_coverage_{VESSEL_TYPE}.parquet"
    if not eu_file.exists():
        logger.warning("  EU ETS coverage not found — skipping")
        return

    eu_df = pd.read_parquet(eu_file)
    eu_df["lat"] = eu_df["h3_id"].apply(lambda h: h3.cell_to_latlng(h)[0])
    eu_df["lon"] = eu_df["h3_id"].apply(lambda h: h3.cell_to_latlng(h)[1])
    eu_df = eu_df[eu_df["lat"].abs() < 72]

    lon_bins = np.arange(-180, 180.5, 0.5)
    lat_bins = np.arange(-72, 72.5, 0.5)
    lon_c = (lon_bins[:-1] + lon_bins[1:]) / 2
    lat_c = (lat_bins[:-1] + lat_bins[1:]) / 2

    eu_full = eu_df[eu_df["weight"] == 1.0]
    eu_partial = eu_df[eu_df["weight"] == 0.5]

    raster_full, _, _ = np.histogram2d(eu_full["lon"], eu_full["lat"],
                                        bins=[lon_bins, lat_bins])
    raster_partial, _, _ = np.histogram2d(eu_partial["lon"], eu_partial["lat"],
                                           bins=[lon_bins, lat_bins])
    raster_full = np.ma.masked_where(raster_full == 0, raster_full)
    raster_partial = np.ma.masked_where(raster_partial == 0, raster_partial)
    raster_partial = np.ma.masked_where(raster_full > 0, raster_partial)

    full_binary = np.zeros(raster_full.shape, dtype=bool)
    full_binary[~raster_full.mask] = True
    labeled, n_features = nd_label(full_binary)
    for comp_id in range(1, n_features + 1):
        if (labeled == comp_id).sum() < 10:
            full_binary[labeled == comp_id] = False
    raster_full = np.ma.masked_where(~full_binary, raster_full.filled(0))
    raster_full = np.ma.masked_where(raster_full == 0, raster_full)
    dilated = binary_dilation(full_binary, iterations=2)
    keep_partial = dilated & (~full_binary)
    raster_partial = np.ma.masked_where(~keep_partial, raster_partial.filled(0))
    raster_partial = np.ma.masked_where(raster_partial == 0, raster_partial)

    LON, LAT = np.meshgrid(lon_c, lat_c)

    fig = plt.figure(figsize=(16, 8))
    ax = setup_basemap(fig)

    # FIX: add graticule labels for richness
    import cartopy.mpl.gridliner as cgrid
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, linestyle=":",
                      color="#8fa3b8", alpha=0.4)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = cgrid.LongitudeFormatter()
    gl.yformatter = cgrid.LatitudeFormatter()
    gl.xlabel_style = {"fontsize": 7, "color": "#666666"}
    gl.ylabel_style = {"fontsize": 7, "color": "#666666"}

    from matplotlib.colors import ListedColormap
    ax.pcolormesh(LON, LAT, raster_full.T, cmap=ListedColormap(["#1F4E79"]),
                  alpha=0.40, shading="auto", transform=ccrs.PlateCarree(),
                  zorder=2, vmin=0, vmax=5)
    ax.pcolormesh(LON, LAT, raster_partial.T, cmap=ListedColormap(["#64B5F6"]),
                  alpha=0.28, shading="auto", transform=ccrs.PlateCarree(),
                  zorder=2, vmin=0, vmax=5)

    paths_file = H3_DIR / f"corridor_h3_paths_{VESSEL_TYPE}.parquet"
    if paths_file.exists():
        h3_paths = pd.read_parquet(paths_file)
        corridors = pd.read_parquet(PROCESSED_DIR / f"corridors_{VESSEL_TYPE}.parquet")
        top50_ids = corridors.nlargest(50, "E_e")["corridor_id"].tolist()
        full_cell_set = set(eu_df.loc[eu_df["weight"] == 1.0, "h3_id"])
        partial_cell_set = set(eu_df.loc[eu_df["weight"] == 0.5, "h3_id"])
        seg_colors = {0: "#0D47A1", 1: "#42A5F5", 2: "#E74C3C"}
        for cid in top50_ids:
            cp = h3_paths[h3_paths["corridor_id"] == cid].sort_values("seq_order")
            if len(cp) < 2:
                continue
            h_ids = cp["h3_id"].values
            lats = cp["h3_id"].apply(lambda h: h3.cell_to_latlng(h)[0]).values
            lons = cp["h3_id"].apply(lambda h: h3.cell_to_latlng(h)[1]).values
            cls = np.full(len(h_ids), 2, dtype=int)
            hs = pd.Series(h_ids)
            cls[hs.isin(partial_cell_set).values] = 1
            cls[hs.isin(full_cell_set).values] = 0
            segments = split_at_antimeridian(lons, lats)
            cum_idx = 0
            for seg_lons, seg_lats in segments:
                seg_len = len(seg_lons)
                seg_cls = cls[cum_idx:cum_idx + seg_len]
                for i in range(seg_len - 1):
                    ax.plot(seg_lons[i:i+2], seg_lats[i:i+2],
                            color=seg_colors[int(seg_cls[i])],
                            alpha=0.65, linewidth=0.8, zorder=5,
                            transform=ccrs.PlateCarree())
                cum_idx += seg_len

    # FIX: add selected port markers with halo for visual richness
    sel_file = RESULTS_DIR / f"selected_ports_{VESSEL_TYPE}.csv"
    if sel_file.exists():
        sel = pd.read_csv(sel_file).dropna(subset=["port_lon", "port_lat"])
        ax.scatter(sel["port_lon"], sel["port_lat"], c="#FF6F00", s=50,
                   alpha=0.12, edgecolors="none", zorder=8,
                   transform=ccrs.PlateCarree())
        ax.scatter(sel["port_lon"], sel["port_lat"], c="#FF6F00", s=8,
                   alpha=0.7, edgecolors="white", linewidths=0.3,
                   zorder=10, transform=ccrs.PlateCarree())

    # FIX: add title annotation
    ax.text(0.015, 0.96, "EU ETS H3 Coverage & Top-50 Corridor Exposure",
            transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    # FIX: add stats annotation
    n_full = int((~raster_full.mask).sum())
    n_partial = int((~raster_partial.mask).sum())
    stats = f"H3 cells: {n_full + n_partial:,}\nFull: {n_full:,} | Partial: {n_partial:,}"
    ax.text(0.985, 0.04, stats, transform=ax.transAxes, fontsize=8.5,
            va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    legend_elements = [
        mpatches.Patch(facecolor="#1F4E79", alpha=0.5,
                       label="EU ETS full coverage (EEA ports + voyages)"),
        mpatches.Patch(facecolor="#64B5F6", alpha=0.4,
                       label="EU ETS partial (50% rule)"),
        Line2D([0], [0], color="#0D47A1", linewidth=2.0,
               label="Segment under full coverage"),
        Line2D([0], [0], color="#42A5F5", linewidth=2.0,
               label="Segment under 50% rule"),
        Line2D([0], [0], color="#E74C3C", linewidth=2.0,
               label="Segment outside EU ETS"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor="#FF6F00",
               markersize=5, label="Selected bunkering port"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=8.5,
              framealpha=0.9, edgecolor="none")

    plt.savefig(REFINE_FIG_DIR / f"fig7_eu_ets_h3_{VESSEL_TYPE}.png", dpi=300,
                bbox_inches="tight", pad_inches=0.1)
    plt.close()
    logger.info("  Saved fig7_eu_ets_h3.")


# ═══════════════════════════════════════════════════════════════
# Fix 10: fig2_siting_map — enrich map, refine port visuals
# ═══════════════════════════════════════════════════════════════

def fig2_siting_map():
    logger.info("[Fig 2] Siting Map (v5 enriched)")
    sel_file = RESULTS_DIR / f"selected_ports_{VESSEL_TYPE}.csv"
    if not sel_file.exists():
        logger.warning("  No selected ports — skipping Fig 2")
        return

    ports = pd.read_parquet(PROCESSED_DIR / f"candidate_ports_{VESSEL_TYPE}.parquet")
    selected = pd.read_csv(sel_file)

    gp_file = POLICY_DIR / f"governance_ports_{VESSEL_TYPE}.parquet"
    camp_map = {}
    if gp_file.exists():
        gp = pd.read_parquet(gp_file)[["port_code", "camp"]]
        camp_map = dict(zip(gp["port_code"], gp["camp"]))
    CAMP_COLORS = {"BRI": "#DAA520", "EU": "#4169E1", "Other": "#8D8D8D"}

    fig = plt.figure(figsize=(16, 8))
    ax = setup_basemap(fig)

    # FIX: add graticule labels
    import cartopy.mpl.gridliner as cgrid
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, linestyle=":",
                      color="#8fa3b8", alpha=0.35)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = cgrid.LongitudeFormatter()
    gl.yformatter = cgrid.LatitudeFormatter()
    gl.xlabel_style = {"fontsize": 7, "color": "#666666"}
    gl.ylabel_style = {"fontsize": 7, "color": "#666666"}
    # FIX: start x-ticks at -150 to reduce empty space on left
    import matplotlib.ticker as mticker
    gl.xlocator = mticker.FixedLocator(np.arange(-150, 181, 30))

    # FIX: add top-20 corridor underlay for visual richness
    paths_file = H3_DIR / f"corridor_h3_paths_{VESSEL_TYPE}.parquet"
    corr_file = PROCESSED_DIR / f"corridors_{VESSEL_TYPE}.parquet"
    if paths_file.exists() and corr_file.exists():
        h3_paths = pd.read_parquet(paths_file)
        corridors = pd.read_parquet(corr_file)
        top20_ids = corridors.nlargest(20, "E_e")["corridor_id"].tolist()
        for cid in top20_ids:
            cp = h3_paths[h3_paths["corridor_id"] == cid].sort_values("seq_order")
            if len(cp) > 1:
                lats = cp["h3_id"].apply(lambda h: h3.cell_to_latlng(h)[0]).values
                lons = cp["h3_id"].apply(lambda h: h3.cell_to_latlng(h)[1]).values
                for seg_lons, seg_lats in split_at_antimeridian(lons, lats):
                    ax.plot(seg_lons, seg_lats, color="#5B7C99", alpha=0.25,
                            linewidth=0.6, zorder=3, transform=ccrs.Geodetic())

    # FIX: candidate ports — larger but more transparent (s=4->6, alpha=0.30->0.12)
    valid = ports.dropna(subset=["port_lon", "port_lat"])
    ax.scatter(valid["port_lon"], valid["port_lat"], c="#B0BEC5", s=6,
               alpha=0.12, edgecolors="none", zorder=3,
               transform=ccrs.PlateCarree(), label="Candidate ports")

    sel_valid = selected.dropna(subset=["port_lon", "port_lat"]).copy()
    if len(sel_valid) > 0:
        for cand in ("n_corridors", "n_corridors_endpoint", "cost_p"):
            if cand in sel_valid.columns:
                rank_col = cand
                break
        else:
            sel_valid["rank_score"] = np.arange(len(sel_valid))
            rank_col = "rank_score"
        sel_valid = sel_valid.sort_values(rank_col, ascending=False)
        sel_valid["camp"] = sel_valid["port_code"].map(camp_map).fillna("Other")

        # FIX: Top-10 hubs — halo + star (larger halo, more transparent)
        hub_ports = sel_valid.head(10)
        for camp, sub in hub_ports.groupby("camp"):
            ax.scatter(sub["port_lon"], sub["port_lat"], c=CAMP_COLORS[camp],
                       s=160, alpha=0.15, edgecolors="none", zorder=10,
                       transform=ccrs.PlateCarree())
            ax.scatter(sub["port_lon"], sub["port_lat"], c=CAMP_COLORS[camp],
                       s=80, marker="*", edgecolors="#1a1a1a", linewidths=0.6,
                       zorder=12, transform=ccrs.PlateCarree())
        ax.scatter([], [], c="#555555", s=80, marker="*", edgecolors="#1a1a1a",
                   linewidths=0.6, label="Top-10 hub ports")

        # FIX: remaining selected — halo + core, more transparent
        rest_ports = sel_valid.iloc[10:]
        for camp, sub in rest_ports.groupby("camp"):
            ax.scatter(sub["port_lon"], sub["port_lat"], c=CAMP_COLORS[camp],
                       s=60, alpha=0.12, edgecolors="none", zorder=9,
                       transform=ccrs.PlateCarree())
            sizes = np.linspace(22, 10, len(sub))
            ax.scatter(sub["port_lon"], sub["port_lat"], c=CAMP_COLORS[camp],
                       s=sizes, edgecolors="#333333", linewidths=0.25, alpha=0.75,
                       zorder=10, transform=ccrs.PlateCarree())
        for camp, col in CAMP_COLORS.items():
            n = int((sel_valid["camp"] == camp).sum())
            ax.scatter([], [], c=col, s=28, edgecolors="#333333", linewidths=0.3,
                       label=f"{camp} camp selected (n={n})")

    # FIX: add title annotation
    ax.text(0.015, 0.96, "MILP Optimal Bunkering Port Siting",
            transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    # FIX: legend moved up slightly to avoid blocking graticule labels
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.03),
              fontsize=8.5, framealpha=0.9, edgecolor="none")

    stats_text = f"Selected: {len(sel_valid)} / {len(valid)} ports\nBudget: 20% of total"
    # FIX: stats moved to upper right to avoid blocking
    ax.text(0.985, 0.96, stats_text, transform=ax.transAxes, fontsize=9,
            ha="right", va="top", bbox=dict(boxstyle="round,pad=0.3",
            facecolor="white", alpha=0.8, edgecolor="none"))

    plt.savefig(REFINE_FIG_DIR / f"fig2_siting_map_{VESSEL_TYPE}.png", dpi=300,
                bbox_inches="tight", pad_inches=0.1)
    plt.close()
    logger.info("  Saved fig2_siting_map.")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("V5 FIGURE LAYOUT FIXES")
    logger.info(f"Output: {REFINE_FIG_DIR}")
    logger.info("=" * 60)

    for name, func in [("fig1_framework", fig1_framework),
                       ("fig8_multiperiod", fig8_multiperiod),
                       ("fig10_bri", fig10_bri),
                       ("fig_imo_carbon", fig_imo_carbon),
                       ("fig11_fueleu", fig11_fueleu),
                       ("fig_E_equity", fig_E_equity),
                       ("fig_C_carbon_cascade", fig_C_carbon_cascade),
                       ("fig3_pareto", fig3_pareto),
                       ("fig7_eu_ets_h3", fig7_eu_ets_h3),
                       ("fig2_siting_map", fig2_siting_map)]:
        try:
            func()
        except Exception as e:
            logger.error(f"  {name} FAILED: {e}", exc_info=True)

    logger.info(f"\nAll v5 figures saved to: {REFINE_FIG_DIR}")


if __name__ == "__main__":
    main()

