"""
Module 03d (v4): Clean Emission Grid for Heatmaps
==================================================
Removes pseudo-inland emission cells from h3_emission_grid: cells whose
center AND all vertices fall on land (outside canal/strait exemption boxes)
come from great-circle fallback attribution (legs without matched AIS
trajectory segments), which cuts straight across continents (e.g. Australia,
Central Asia). Coastal cells (>=1 sea vertex), lake/sea-surface cells and
canal/strait cells are kept, so genuine inland-water heat (rivers/lakes)
remains visible. The filter rule is identical to 03b's path-graph filter.

Outputs (02_数据_output/h3_grid_v4/):
  h3_emission_grid_clean_{vt}.parquet
"""

import sys
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import h3

sys.path.insert(0, str(Path(__file__).parent))
from config import H3_DIR, VESSEL_TYPE
import importlib
m03b = importlib.import_module("03b_h3_corridor_paths")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_clean_grid(vessel_type: str):
    t0 = time.time()
    src = H3_DIR / f"h3_emission_grid_{vessel_type}.parquet"
    if not src.exists():
        raise RuntimeError(f"H3 grid not found at {src}")
    h3_df = pd.read_parquet(src)
    logger.info(f"  Loaded H3 grid: {len(h3_df):,} cells, "
                f"{h3_df['emission_tco2e'].sum()/1e6:.2f} Mt")

    mask, mask_step = m03b.build_land_mask(0.25)
    em_keep, n_drop = m03b.filter_emission_cells(h3_df, mask, mask_step)
    clean = h3_df[h3_df["h3_id"].isin(set(em_keep))].copy()
    clean = clean.reset_index(drop=True)

    out = H3_DIR / f"h3_emission_grid_clean_{vessel_type}.parquet"
    clean.to_parquet(out, index=False)
    logger.info(f"  Clean grid: {len(clean):,} cells "
                f"({n_drop:,} inland pseudo-cells dropped), "
                f"{clean['emission_tco2e'].sum()/1e6:.2f} Mt "
                f"({time.time()-t0:.0f}s)")
    logger.info(f"  Saved: {out.name}")
    return clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vessel-type", default=VESSEL_TYPE)
    args = parser.parse_args()
    build_clean_grid(args.vessel_type)
