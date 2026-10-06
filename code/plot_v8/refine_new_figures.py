"""
New Figures for TRE-09MEET — Final Version
============================================
Three figures added to address evaluation findings:

  Fig A: fig_siting_comparison — Three-way port siting comparison (M^G vs M^EU vs M^BRI)
  Fig B: fig_overlap_bri_eu    — BRI-EU ETS overlap (map + inset Euler diagram)
  Fig C: fig_ets_recycling    — EU ETS revenue recycling budget decomposition

Style: matches refine_v2_gov.py (Times New Roman, STIX math, 300 DPI)

Usage:
  python refine_new_figures.py            # generate all three
  python refine_new_figures.py --fig siting   # generate one

Dependencies:
  matplotlib, numpy, pandas, cartopy, h3, pyarrow
  Project data directory: 02_数据_output/
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyBboxPatch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import h3

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repository root
OUTPUT_DIR = PROJECT_ROOT / "data" / "derived"
CODE_DIR = PROJECT_ROOT / "code" / "calc_v5"
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_DIR = OUTPUT_DIR / "processed_v5"
H3_DIR = OUTPUT_DIR / "h3_grid_v5"
RESULTS_DIR = OUTPUT_DIR / "milp_results_v5"
POLICY_DIR = OUTPUT_DIR / "policy_analysis_v5"
POLICY_DIR_V2 = OUTPUT_DIR / "policy_analysis_v5"

VESSEL_TYPE = "container"

# ============================================================
# Academic style (matches refine_v2_gov.py)
# ============================================================
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
CAMP_COLORS = {"BRI": "#DAA520", "EU": "#4169E1", "Other": "#8D8D8D"}
GOV_COLORS = {
    "EU_ETS": "#4169E1",
    "BRI_only": "#DAA520",
    "FuelEU_exposed": "#8E44AD",
    "Vacuum": "#E74C3C",
}


# ============================================================
# Helpers
# ============================================================
def setup_basemap(fig, position, extent=None):
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
    """Split a polyline at the antimeridian to avoid horizontal streaks."""
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
    path = FIG_DIR / f"{name}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name} ({path.stat().st_size / 1024:.0f} KB)")


# ============================================================
# Fig A: Three-way port siting comparison (M^G vs M^EU vs M^BRI)
# ============================================================
def fig_siting_comparison():
    """
    Three-panel world map comparing the optimal bunkering port selection
    under each rule-competition model.
    """
    print("[Fig A] Three-way port siting comparison")

    # Load port coordinates and camp from governance_ports
    gov_ports = pd.read_parquet(POLICY_DIR / f"governance_ports_{VESSEL_TYPE}.parquet")
    port_info = gov_ports[["port_code", "port_lon", "port_lat", "camp"]].drop_duplicates("port_code")
    port_lookup = port_info.set_index("port_code")

    # Load M^G selected ports
    mg = pd.read_csv(RESULTS_DIR / f"selected_ports_{VESSEL_TYPE}.csv")
    mg_codes = set(mg["port_code"].tolist())

    # Load M^EU selected ports
    meu = pd.read_csv(POLICY_DIR / f"meU_selected_ports_{VESSEL_TYPE}.csv")
    meu_codes = set(meu["port_code"].tolist())

    # Load M^BRI selected ports
    mbri = pd.read_csv(POLICY_DIR / f"meB_selected_ports_{VESSEL_TYPE}.csv")
    mbri_codes = set(mbri["port_code"].tolist())

    # All candidate ports (grey background)
    all_ports = port_info.copy()

    # Load activated corridors for M^G
    corr_g = pd.read_csv(RESULTS_DIR / f"activated_corridors_{VESSEL_TYPE}.csv")

    # Load corridor paths for map drawing
    paths_file = H3_DIR / f"corridor_h3_paths_{VESSEL_TYPE}.parquet"
    paths = pd.read_parquet(paths_file) if paths_file.exists() else None
    path_lookup = {}
    if paths is not None:
        path_lookup = paths.groupby("corridor_id")["h3_id"].apply(list).to_dict()

    # Load governance corridors for model-specific corridor selection
    gov_corr = pd.read_parquet(POLICY_DIR / f"governance_corridors_{VESSEL_TYPE}.parquet")

    models = [
        ("M$^G$ (global planner)", mg_codes, "#2e7d32", 211, 100.43, 2758),
        ("M$^{EU}$ (EU rule sphere)", meu_codes, "#4169E1", 182, 22.17, 1025),
        ("M$^{BRI}$ (BRI investment domain)", mbri_codes, "#DAA520", 228, 42.64, 2352),
    ]

    fig = plt.figure(figsize=(16, 14))

    for i, (title, codes, accent, n_ports, abate, n_corr) in enumerate(models):
        ax = setup_basemap(fig, 311 + i)

        # Draw activated corridors (top 30 by emission)
        if paths is not None:
            if i == 0:
                # M^G: use actual activated corridors
                top_corr = corr_g.nlargest(30, "E_e") if "E_e" in corr_g.columns else corr_g.head(30)
            elif i == 1:
                # M^EU: draw EU-sphere corridors
                top_corr = gov_corr[gov_corr["gov_type"] == "EU_ETS"].nlargest(30, "E_e")
            else:
                # M^BRI: draw BRI corridors
                top_corr = gov_corr[gov_corr["gov_type"] == "BRI_only"].nlargest(30, "E_e")

            for _, row in top_corr.iterrows():
                cid = row.get("corridor_id", row.name)
                cells = path_lookup.get(cid)
                if cells and len(cells) >= 2:
                    lats = [h3.cell_to_latlng(h)[0] for h in cells]
                    lons = [h3.cell_to_latlng(h)[1] for h in cells]
                    for sl, sa in split_at_antimeridian(lons, lats):
                        ax.plot(sl, sa, color=accent, linewidth=0.8, alpha=0.3,
                                zorder=5, transform=ccrs.Geodetic())

        # Plot all candidate ports (grey, small)
        ax.scatter(all_ports["port_lon"], all_ports["port_lat"], c="#cccccc", s=2,
                   alpha=0.25, zorder=3, transform=ccrs.PlateCarree())

        # Plot selected ports colored by camp
        sel_ports = port_info[port_info["port_code"].isin(codes)]
        for camp, color in CAMP_COLORS.items():
            sub = sel_ports[sel_ports["camp"] == camp]
            if len(sub) == 0:
                continue
            ax.scatter(sub["port_lon"], sub["port_lat"], s=18, c=color,
                       alpha=0.75, edgecolors="white", linewidths=0.3, zorder=6,
                       transform=ccrs.PlateCarree(), label=f"{camp} ({len(sub)})")

        # Annotate stats
        ax.text(0.99, 0.97, f"{title}\n{n_ports} ports | {abate:.2f} Mt | {n_corr} corridors",
                transform=ax.transAxes, fontsize=10.5, fontweight="bold",
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
        panel_label(ax, f"({chr(97 + i)})")
        ax.legend(loc="lower left", fontsize=10, framealpha=0.9, ncol=3)

    plt.tight_layout(h_pad=0.5)
    save(f"fig_siting_comparison_{VESSEL_TYPE}")


# ============================================================
# Fig B: BRI-EU ETS overlap (map + inset Euler diagram)
# ============================================================
def fig_overlap():
    """
    Single-panel world map of BRI-EU overlap corridors with an inset
    Euler diagram in the lower-left corner.
    """
    print("[Fig B] BRI-EU ETS overlap")
    gov = pd.read_parquet(POLICY_DIR / f"governance_corridors_{VESSEL_TYPE}.parquet")

    both = gov[(gov["is_bri"]) & (gov["eu_ets_endpoint"])]
    bri_only = gov[(gov["is_bri"]) & (~gov["eu_ets_endpoint"])]
    eu_only = gov[(~gov["is_bri"]) & (gov["eu_ets_endpoint"])]
    neither = gov[(~gov["is_bri"]) & (~gov["eu_ets_endpoint"])]
    em_total = gov["E_e"].sum()

    # --- Single figure: map with inset Euler diagram ---
    fig = plt.figure(figsize=(14, 7.5))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -56, 72], crs=ccrs.PlateCarree())
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

    # Draw overlap corridors on map
    paths_file = H3_DIR / f"corridor_h3_paths_{VESSEL_TYPE}.parquet"
    paths = pd.read_parquet(paths_file) if paths_file.exists() else None
    path_lookup = {}
    if paths is not None:
        path_lookup = paths.groupby("corridor_id")["h3_id"].apply(list).to_dict()

    overlap_colors = {"both": "#7B3F00", "bri": "#DAA520", "eu": "#4169E1"}

    for label, df, color_key in [
        ("BRI $\\cap$ ETS", both, "both"),
        ("BRI only", bri_only, "bri"),
        ("EU ETS only", eu_only, "eu"),
    ]:
        top = df.nlargest(15, "E_e")
        for _, row in top.iterrows():
            cid = row["corridor_id"]
            cells = path_lookup.get(cid)
            if cells is None or len(cells) < 2:
                continue
            lats = [h3.cell_to_latlng(h)[0] for h in cells]
            lons = [h3.cell_to_latlng(h)[1] for h in cells]
            for sl, sa in split_at_antimeridian(lons, lats):
                ax.plot(sl, sa, color=overlap_colors[color_key], linewidth=1.5, alpha=0.7,
                        zorder=6, transform=ccrs.Geodetic())

    # --- Inset Euler diagram: compact, positioned in map's lower-left ---
    inset_ax = fig.add_axes([0.09, 0.22, 0.24, 0.33])
    inset_ax.set_xlim(-2.7, 2.7)
    inset_ax.set_ylim(-2.6, 2.6)
    inset_ax.set_aspect("equal")
    inset_ax.axis("off")

    # Tight background box with minimal padding
    bg = FancyBboxPatch((-2.65, -2.55), 5.0, 5.1, boxstyle="round,pad=0.06",
                       facecolor="white", alpha=0.7, edgecolor="#888888",
                       linewidth=0.7, zorder=20)
    inset_ax.add_patch(bg)

    r = 1.3
    d = 0.6
    c_bri = Circle((-d, 0.25), r, facecolor="#DAA520", alpha=0.30,
                   edgecolor="#DAA520", linewidth=1.8, zorder=22)
    c_eu = Circle((d, 0.25), r, facecolor="#4169E1", alpha=0.30,
                  edgecolor="#4169E1", linewidth=1.8, zorder=22)
    inset_ax.add_patch(c_bri)
    inset_ax.add_patch(c_eu)

    # Labels — compact font, positioned tight to circles
    inset_ax.text(-1.45, 0.5,
                  f"BRI only\n{len(bri_only)} corr.\n{bri_only['E_e'].sum()/1e6:.1f} Mt\n"
                  f"({bri_only['E_e'].sum()/em_total*100:.1f}%)",
                  ha="center", va="center", fontsize=7, fontweight="bold",
                  color="#8B6914", zorder=25)
    inset_ax.text(1.45, 0.5,
                  f"EU ETS only\n{len(eu_only)} corr.\n{eu_only['E_e'].sum()/1e6:.1f} Mt\n"
                  f"({eu_only['E_e'].sum()/em_total*100:.1f}%)",
                  ha="center", va="center", fontsize=7, fontweight="bold",
                  color="#1E3A8A", zorder=25)
    inset_ax.text(0, -0.1,
                  f"BRI $\\cap$ ETS\n{len(both)} corr.\n{both['E_e'].sum()/1e6:.1f} Mt\n"
                  f"({both['E_e'].sum()/em_total*100:.1f}%)",
                  ha="center", va="center", fontsize=7, fontweight="bold",
                  color="#7B3F00", zorder=25)
    inset_ax.text(0, -1.95,
                  f"Neither\n{len(neither)} corr.\n{neither['E_e'].sum()/1e6:.1f} Mt\n"
                  f"({neither['E_e'].sum()/em_total*100:.1f}%)",
                  ha="center", va="center", fontsize=7, fontweight="bold",
                  color="#666666", zorder=25)

    # Circle title labels
    inset_ax.text(-d, 1.65, "BRI\n(59 states)", ha="center", va="bottom",
                  fontsize=7, fontweight="bold", color="#DAA520", zorder=25)
    inset_ax.text(d, 1.65, "EU ETS\nendpoint", ha="center", va="bottom",
                  fontsize=7, fontweight="bold", color="#4169E1", zorder=25)

    # --- Legend at bottom center ---
    legend_patches = [
        mpatches.Patch(color=overlap_colors["both"], label="BRI $\\cap$ ETS overlap"),
        mpatches.Patch(color=overlap_colors["bri"], label="BRI only"),
        mpatches.Patch(color=overlap_colors["eu"], label="EU ETS only"),
    ]
    ax.legend(handles=legend_patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=10,
              framealpha=0.9, frameon=True)

    save(f"fig_overlap_bri_eu_{VESSEL_TYPE}")


# ============================================================
# Fig C: EU ETS revenue recycling
# ============================================================
def fig_ets_recycling():
    """
    Two-panel figure showing (a) budget decomposition + abatement and
    (b) mismatch cost + port changes across gamma levels.
    """
    print("[Fig C] EU ETS revenue recycling")

    # Data from policy_analysis_v5 (matches paper numbers)
    gamma_data = [
        {"gamma": 0.0, "ets_rev": 0.0, "abate": 86.54, "mpc": 13.83,
         "ports": 190, "added": 78, "removed": 99},
        {"gamma": 0.5, "ets_rev": 966.73, "abate": 98.25, "mpc": 2.17,
         "ports": 259, "added": 79, "removed": 31},
        {"gamma": 1.0, "ets_rev": 1967.86, "abate": 103.55, "mpc": -3.11,
         "ports": 322, "added": 111, "removed": 0},
    ]
    base_budget = 3070.2  # M USD/yr
    global_opt = 100.43    # Mt (M^G benchmark)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    gammas = [d["gamma"] for d in gamma_data]
    x = np.arange(len(gammas))
    width = 0.35

    # (a) Budget decomposition + abatement
    ax1_twin = ax1.twinx()
    bars1 = ax1.bar(x - width/2, [base_budget]*3, width, color="#4169E1", alpha=0.7,
                     edgecolor="black", linewidth=0.4, label="Base infrastructure budget")
    bars2 = ax1.bar(x - width/2, [d["ets_rev"] for d in gamma_data], width,
                     bottom=[base_budget]*3,
                     color="#DAA520", alpha=0.7, edgecolor="black", linewidth=0.4,
                     label="ETS revenue recycled")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"$\\gamma={g}$" for g in gammas])
    ax1.set_ylabel("Budget (M USD yr$^{-1}$)", fontsize=11)
    ax1.set_ylim(0, 5600)

    # Abatement line
    abate_vals = [d["abate"] for d in gamma_data]
    ax1_twin.plot(x, abate_vals, "o-", color="#2e7d32", linewidth=2.5,
                  markersize=10, zorder=5, label="Global abatement")
    ax1_twin.axhline(global_opt, color="#E74C3C", linestyle="--", linewidth=1.2,
                     label=f"$M^G$ benchmark ({global_opt:.2f} Mt)")
    ax1_twin.set_ylabel("Abatement (Mt CO$_2$e yr$^{-1}$)", fontsize=11)
    ax1_twin.set_ylim(0, 115)

    for xi, d in zip(x, gamma_data):
        ax1.annotate(f"{d['ets_rev']:.0f}",
                     xy=(xi - width/2, base_budget + d['ets_rev']/2),
                     ha="center", va="center", fontsize=8, fontweight="bold",
                     color="white" if d['ets_rev'] > 100 else "#DAA520")
        ax1_twin.annotate(f"{d['abate']:.2f}", xy=(xi, d['abate']),
                         xytext=(-10, -14), textcoords="offset points",
                         ha="right", fontsize=9, fontweight="bold",
                         color="#2e7d32")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
               ncol=2, fontsize=8, framealpha=0.9)
    ax1.set_xlabel("ETS revenue recycling share $\\gamma$")
    ax1.grid(axis="y", alpha=0.3)
    panel_label(ax1, "(a)")

    # (b) Mismatch cost and port changes
    mpc_vals = [d["mpc"] for d in gamma_data]
    added_vals = [d["added"] for d in gamma_data]
    removed_vals = [-d["removed"] for d in gamma_data]

    ax2.bar(x - 0.2, mpc_vals, 0.4, color="#E74C3C", alpha=0.7,
            edgecolor="black", linewidth=0.4, label="MPC (%)")
    ax2.set_ylabel("Policy mismatch cost MPC (%)", fontsize=11, color="#E74C3C")
    ax2.tick_params(axis="y", labelcolor="#E74C3C")
    ax2.set_ylim(-5, 16)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"$\\gamma={g}$" for g in gammas])
    ax2.set_xlabel("ETS revenue recycling share $\\gamma$")

    for xi, mpc in zip(x - 0.2, mpc_vals):
        ax2.annotate(f"{mpc:.1f}%", xy=(xi, mpc), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=9,
                     fontweight="bold", color="#E74C3C")

    ax2_twin = ax2.twinx()
    ax2_twin.bar(x + 0.2, added_vals, 0.35, color="#2e7d32", alpha=0.6,
                  edgecolor="black", linewidth=0.4, label="Ports added")
    ax2_twin.bar(x + 0.2, removed_vals, 0.35, color="#888888", alpha=0.6,
                  edgecolor="black", linewidth=0.4, label="Ports removed")
    ax2_twin.set_ylabel("Port count change", fontsize=11)
    ax2_twin.set_ylim(-120, 130)
    ax2_twin.axhline(0, color="black", linewidth=0.5)

    lines_l, labels_l = ax2.get_legend_handles_labels()
    lines_r, labels_r = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines_l + lines_r, labels_l + labels_r, loc="upper center",
               fontsize=8, framealpha=0.95, bbox_to_anchor=(0.5, 0.98))
    ax2.grid(axis="y", alpha=0.3)
    panel_label(ax2, "(b)")

    plt.tight_layout()
    save(f"fig_ets_recycling_{VESSEL_TYPE}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    target = None
    if "--fig" in sys.argv:
        idx = sys.argv.index("--fig")
        if idx + 1 < len(sys.argv):
            target = sys.argv[idx + 1].lower()

    print("=" * 60)
    print("NEW FIGURES FOR TRE-09MEET (v7_revision)")
    print("=" * 60)

    if target in (None, "all"):
        fig_siting_comparison()
        fig_overlap()
        fig_ets_recycling()
    elif target in ("siting", "a"):
        fig_siting_comparison()
    elif target in ("overlap", "b"):
        fig_overlap()
    elif target in ("recycling", "ets", "c"):
        fig_ets_recycling()
    else:
        print(f"Unknown figure: {target}")
        print("Usage: python refine_new_figures.py [--fig siting|overlap|recycling]")
        sys.exit(1)

    print("Done.")

