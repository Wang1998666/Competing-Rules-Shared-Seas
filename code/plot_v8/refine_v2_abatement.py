"""
Figure Refinement — Abatement Heatmaps (Scenario × Period) — v5 data
=====================================================================
v5 copy of plot_v6final/refine_v2_abatement.py: reads the corrected
(C1-C4) emission pipeline outputs (processed_v5 / h3_grid_v5 /
milp_results_v5 / policy_analysis_v5) and writes to v7_revision/.
The original v6_final figures are NOT modified.

  fig_abate_matrix    : scenario × period abatement matrix (Mt, heatmap)
  fig_abate_evolution : M^G six-period (2025-2050) abatement heat maps (3x2)
  fig_abate_scenario  : scenario comparison at 20% budget (M^G/M^EU/M^BRI/
                        FuelEU) abatement heat maps (2x2)
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import h3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repository root
OUTPUT_DIR = PROJECT_ROOT / "data" / "derived"
REFINE_FIG_DIR = PROJECT_ROOT / "figures"
REFINE_FIG_DIR.mkdir(parents=True, exist_ok=True)

CODE_DIR = PROJECT_ROOT / "code" / "calc_v5"
sys.path.insert(0, str(CODE_DIR))
import importlib
m03 = importlib.import_module("03_h3_attribution")

H3_DIR = OUTPUT_DIR / "h3_grid_v5"
PROCESSED_DIR = OUTPUT_DIR / "processed_v5"
RESULTS_DIR = OUTPUT_DIR / "milp_results_v5"
POLICY_V1 = OUTPUT_DIR / "policy_analysis_v5"
POLICY_V3 = OUTPUT_DIR / "policy_analysis_v5"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

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
OCEAN_COLOR = "#dceef7"
LAND_COLOR = "#f0ede8"
COAST_COLOR = "#999999"

PERIODS = [2025, 2030, 2035, 2040, 2045, 2050]
FUEL = "green_methanol"
ETA_COL = f"eta_{FUEL}"


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


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_corridor_abatement(corr_file, corridors):
    """abatement per activated corridor = E_e × eta × alpha (Mt).
    Accepts a CSV path or an already-loaded DataFrame; handles files that
    only carry corridor_id by merging the corridor attributes table."""
    act = (pd.read_csv(corr_file) if isinstance(corr_file, (str, Path))
           else corr_file.copy())
    if "abatement_tco2e" in act.columns:
        return act[["corridor_id", "abatement_tco2e"]].rename(
            columns={"abatement_tco2e": "abatement"})
    if "corridor_id" not in act.columns:
        raise KeyError(f"{corr_file.name}: no corridor_id column")
    act = act[["corridor_id"]].merge(
        corridors[["corridor_id", "E_e", ETA_COL, "alpha_e"]],
        on="corridor_id", how="left")
    act["abatement"] = (act["E_e"].fillna(0) * act[ETA_COL].fillna(0)
                         * act["alpha_e"].fillna(0))
    return act[["corridor_id", "abatement"]]


def abatement_to_cells(act, paths, emit_mode="length", scale_to=None):
    """Distribute corridor abatement over its v5 path cells (length-weighted).
    If scale_to is given (total abatement in tCO2e, e.g. the MILP model value
    that accounts for fuel-availability windows), cell values are rescaled so
    the grid sum equals it exactly. Returns DataFrame(h3_id, abatement_tco2e)."""
    path_lookup = paths.groupby("corridor_id")["h3_id"].apply(list).to_dict()
    rows = []
    for cid, ab in zip(act["corridor_id"], act["abatement"]):
        cells = path_lookup.get(cid)
        if cells is None or len(cells) < 2 or ab <= 0:
            continue
        lats = np.array([h3.cell_to_latlng(c)[0] for c in cells])
        lons = np.array([h3.cell_to_latlng(c)[1] for c in cells])
        seg_len = haversine_km(lons[:-1], lats[:-1], lons[1:], lats[1:])
        tot = seg_len.sum()
        if tot <= 0:
            continue
        mid_cells = cells[:-1]  # emission of segment k → cell k
        seg_ab = ab * seg_len / tot
        for c, e in zip(mid_cells, seg_ab):
            rows.append((c, e))
    if not rows:
        return pd.DataFrame(columns=["h3_id", "abatement_tco2e"])
    df = pd.DataFrame(rows, columns=["h3_id", "abatement_tco2e"])
    df = df.groupby("h3_id", as_index=False)["abatement_tco2e"].sum()
    if scale_to is not None and df["abatement_tco2e"].sum() > 0:
        df["abatement_tco2e"] *= scale_to / df["abatement_tco2e"].sum()
    return df


def rasterize(ab_cells, grid=0.5):
    """0.5° raster of cell-level abatement (t per cell)."""
    lon_bins = np.arange(-180, 180.5, grid)
    lat_bins = np.arange(-72, 72.5, grid)
    raster, _, _ = np.histogram2d(ab_cells["lon"], ab_cells["lat"],
                                  bins=[lon_bins, lat_bins],
                                  weights=ab_cells["abatement_tco2e"])
    raster = np.ma.masked_where(raster == 0, raster)
    lon_c = (lon_bins[:-1] + lon_bins[1:]) / 2
    lat_c = (lat_bins[:-1] + lat_bins[1:]) / 2
    LON, LAT = np.meshgrid(lon_c, lat_c)
    return raster, LON, LAT


def kde_rasterize(ab_cells, grid=0.5, sigma=2.0):
    """Kernel-density-style abatement field: rasterize the per-cell abatement
    and smooth it with a Gaussian kernel (bandwidth ~sigma×grid degrees), so
    the heat appears as broad density bands rather than thin path lines.
    Pixels whose centers fall on land are masked (sea-constrained domain).
    Returns (raster, LON, LAT) with values in t per grid cell."""
    lon_bins = np.arange(-180, 180.5, grid)
    lat_bins = np.arange(-72, 72.5, grid)
    raster, _, _ = np.histogram2d(ab_cells["lon"], ab_cells["lat"],
                                  bins=[lon_bins, lat_bins],
                                  weights=ab_cells["abatement_tco2e"])
    # Gaussian smoothing (kernel-density approximation)
    from scipy.ndimage import gaussian_filter
    raster = gaussian_filter(raster, sigma=sigma)
    # sea-constrained domain: drop pixels whose center is on land (0.25° mask)
    mask = m03._get_land_mask()
    lon_c = (lon_bins[:-1] + lon_bins[1:]) / 2
    lat_c = (lat_bins[:-1] + lat_bins[1:]) / 2
    LON, LAT = np.meshgrid(lon_c, lat_c)
    i_idx = np.clip(((LON + 180) / 0.25).astype(int), 0, mask.shape[0] - 1)
    j_idx = np.clip(((LAT + 90) / 0.25).astype(int), 0, mask.shape[1] - 1)
    land = mask[i_idx, j_idx]  # shape (n_lat, n_lon)
    raster = np.ma.masked_where((raster <= 0) | land.T, raster)
    return raster, LON, LAT


def setup_map_ax(fig, nrows, ncols, idx, extent=(-180, 180, -62, 75)):
    ax = fig.add_subplot(nrows, ncols, idx, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '110m',
                   facecolor=OCEAN_COLOR), zorder=0)
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m',
                   facecolor=LAND_COLOR), zorder=0)
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'coastline', '110m',
                   facecolor='none', edgecolor=COAST_COLOR, linewidth=0.4), zorder=1)
    return ax


def cell_df_with_lonlat(ab_cells):
    df = ab_cells.copy()
    df["lat"] = df["h3_id"].apply(lambda h: h3.cell_to_latlng(h)[0])
    df["lon"] = df["h3_id"].apply(lambda h: h3.cell_to_latlng(h)[1])
    return df


# ═══════════════════════════════════════════════════════════════
# Fig 1: Scenario × Period abatement matrix
# ═══════════════════════════════════════════════════════════════

def fig_abate_matrix(corridors):
    logger.info("[Abate Matrix] scenario × period")
    # M^G multi-period
    mp_sum = pd.read_csv(RESULTS_DIR / "multiperiod_summary_container.csv")
    mg_series = mp_sum.set_index("period")["period_abatement_tco2e"] / 1e6

    # single-period scenarios @20% budget
    scen = {
        "M$^G$ (20%)": load_corridor_abatement(
            RESULTS_DIR / "activated_corridors_container.csv", corridors),
        "M$^{EU}$ (20%)": load_corridor_abatement(
            POLICY_V1 / "fueleu_activated_corridors_container_vieweu_b20_g0.0.csv",
            corridors),
        "M$^{BRI}$ (20%)": load_corridor_abatement(
            POLICY_V1 / "bri_view_activated_corridors_container_b20.csv",
            corridors),
        "FuelEU (20%)": load_corridor_abatement(
            POLICY_V3 / "fueleu_activated_corridors_container.csv", corridors),
    }
    rows = ["M$^G$ (multi-period)"] + list(scen.keys())
    mat = np.zeros((len(rows), len(PERIODS)))
    mat[0] = [mg_series.get(p, 0.0) for p in PERIODS]
    for i, (name, df) in enumerate(scen.items(), start=1):
        v = df["abatement"].sum() / 1e6
        mat[i] = v  # single-period annual value (constant across columns)

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto",
                   norm=LogNorm(vmin=max(mat[mat > 0].min(), 0.01),
                                vmax=mat.max()))
    ax.set_xticks(range(len(PERIODS)))
    ax.set_xticklabels(PERIODS)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xlabel("Period (year)")
    for i in range(len(rows)):
        for j in range(len(PERIODS)):
            v = mat[i, j]
            if v > 0:
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9,
                        color="white" if v > mat.max() * 0.4 else "black",
                        fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Annual abatement (Mt CO$_2$e)", fontsize=10)
    ax.set_title("Abatement by Scenario and Period (green methanol)", fontsize=11)
    # annotate single-period rows
    for i, name in enumerate(scen.keys(), start=1):
        ax.text(len(PERIODS) - 0.5, i, "single-period\n@20% budget",
                ha="center", va="center", fontsize=6.5, color="#666666")
    plt.tight_layout()
    plt.savefig(REFINE_FIG_DIR / "fig_abate_matrix_container.png", dpi=300)
    plt.close()
    logger.info("  Saved fig_abate_matrix.")


# ═══════════════════════════════════════════════════════════════
# Fig 2: M^G six-period abatement heat maps (3x2)
# ═══════════════════════════════════════════════════════════════

def fig_abate_evolution(corridors, paths):
    logger.info("[Abate Evolution] M^G 2025-2050")
    fig = plt.figure(figsize=(16, 10))
    mp_sum = pd.read_csv(RESULTS_DIR / "multiperiod_summary_container.csv")
    period_abate = mp_sum.set_index("period")["period_abatement_tco2e"].to_dict()
    for idx, p in enumerate(PERIODS):
        mp = pd.read_csv(RESULTS_DIR / f"multiperiod_corridors_{p}_container.csv")
        act = load_corridor_abatement(mp, corridors)
        # rescale spatial distribution to the MILP model value (accounts for
        # fuel-availability windows, learning, adoption)
        ab_cells = abatement_to_cells(act, paths,
                                      scale_to=period_abate.get(p, 0.0))
        total_mt = period_abate.get(p, 0.0) / 1e6
        if len(ab_cells) == 0:
            logger.warning(f"  period {p}: no cells")
            continue
        ab_cells = cell_df_with_lonlat(ab_cells)
        raster, LON, LAT = kde_rasterize(ab_cells, sigma=0.5)
        pos = idx + 1
        ax = setup_map_ax(fig, 3, 2, pos)
        pos_vals = raster.compressed()
        if len(pos_vals) == 0:
            continue
        norm = LogNorm(vmin=max(np.percentile(pos_vals, 10), 0.1),
                       vmax=np.percentile(pos_vals, 99))
        ax.pcolormesh(LON, LAT, raster.T, cmap="BuGn", norm=norm,
                      alpha=0.9, shading="auto", transform=ccrs.PlateCarree(), zorder=2)
        ax.text(0.02, 0.95, f"({chr(97 + idx)})  {p}   {total_mt:.2f} Mt",
                transform=ax.transAxes, fontsize=11, fontweight="bold", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
    fig.suptitle("M$^G$ Multi-Period Abatement Density (KDE-smoothed, green methanol)",
                 fontsize=13, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(REFINE_FIG_DIR / "fig_abate_evolution_container.png", dpi=300)
    plt.close()
    logger.info("  Saved fig_abate_evolution.")


# ═══════════════════════════════════════════════════════════════
# Fig 3: Scenario comparison abatement heat maps (2x2)
# ═══════════════════════════════════════════════════════════════

def fig_abate_scenario(corridors, paths):
    logger.info("[Abate Scenario] M^G / M^EU / M^BRI / FuelEU @20%")
    scenarios = [
        ("M$^G$ (global)", RESULTS_DIR / "activated_corridors_container.csv"),
        ("M$^{EU}$ (EU sphere)", POLICY_V1 / "fueleu_activated_corridors_container_vieweu_b20_g0.0.csv"),
        ("M$^{BRI}$ (BRI domain)", POLICY_V1 / "bri_view_activated_corridors_container_b20.csv"),
        ("FuelEU forcing", POLICY_V3 / "fueleu_activated_corridors_container.csv"),
    ]
    fig = plt.figure(figsize=(16, 9.5))
    vmax_all = 0.0
    rasters = []
    for name, f in scenarios:
        act = load_corridor_abatement(f, corridors)
        ab_cells = abatement_to_cells(act, paths)
        if len(ab_cells):
            ab_cells = cell_df_with_lonlat(ab_cells)
            raster, _, _ = kde_rasterize(ab_cells, sigma=0.5)
            rasters.append((name, raster, act["abatement"].sum() / 1e6))
            if raster.count() > 0:
                vmax_all = max(vmax_all, float(raster.max()))
    for idx, (name, raster, total_mt) in enumerate(rasters):
        ax = setup_map_ax(fig, 2, 2, idx + 1)
        pos_vals = raster.compressed()
        if len(pos_vals) == 0:
            continue
        norm = LogNorm(vmin=max(np.percentile(pos_vals, 10), 0.1),
                       vmax=vmax_all)
        ax.pcolormesh(*rasterize_geo(), raster.T, cmap="BuGn", norm=norm,
                      alpha=0.9, shading="auto", transform=ccrs.PlateCarree(), zorder=2)
        ax.text(0.02, 0.95, f"({chr(97 + idx)})  {name}   {total_mt:.1f} Mt",
                transform=ax.transAxes, fontsize=11, fontweight="bold", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
    fig.suptitle("Abatement Density under Competing Rules (KDE-smoothed, 20% budget,"
                 " green methanol)", fontsize=13, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(REFINE_FIG_DIR / "fig_abate_scenario_container.png", dpi=300)
    plt.close()
    logger.info("  Saved fig_abate_scenario.")


_GEO_CACHE = {}


def rasterize_geo():
    if "geo" not in _GEO_CACHE:
        lon_bins = np.arange(-180, 180.5, 0.5)
        lat_bins = np.arange(-72, 72.5, 0.5)
        lon_c = (lon_bins[:-1] + lon_bins[1:]) / 2
        lat_c = (lat_bins[:-1] + lat_bins[1:]) / 2
        _GEO_CACHE["geo"] = (lon_bins, lat_bins, np.meshgrid(lon_c, lat_c))
    return _GEO_CACHE["geo"][2]


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    corridors = pd.read_parquet(PROCESSED_DIR / "corridors_container.parquet")
    paths = pd.read_parquet(H3_DIR / "corridor_h3_paths_container.parquet")
    logger.info(f"  corridors {len(corridors):,}, path rows {len(paths):,}")
    fig_abate_matrix(corridors)
    fig_abate_evolution(corridors, paths)
    fig_abate_scenario(corridors, paths)
    logger.info("Done.")


if __name__ == "__main__":
    main()
