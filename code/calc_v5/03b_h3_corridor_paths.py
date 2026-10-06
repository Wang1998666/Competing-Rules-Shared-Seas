"""
Module 03b (v4): H3 Corridor Paths — AIS-Cell-Only Routing Network
==================================================================
Root-fix version per user directive: the routing graph contains ONLY the
H3 cells that AIS vessels actually visited (emission cells). Cells not
visited are treated as land (not traversable) — no ocean-completion, no
bridge cells, no exemption boxes (the v4 intermediate scheme is dropped).

Every corridor is routed with a single-source Dijkstra grouped by origin
port (one run serves all corridors sharing that origin). Corridors whose
endpoints are unreachable on the AIS-cell network, or whose endpoint has
no coordinates (placeholder "AAAAA"), are SKIPPED (no path → no line drawn;
no GC fallback is ever used).

Edge weight favors high-emission lanes: w = 1/(log1p(emission)+0.01).

Outputs (02_数据_output/h3_grid_v4/):
  corridor_h3_paths_{vt}.parquet, h3_corridor_coverage_{vt}.parquet
"""

import sys
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import h3
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PROCESSED_DIR, H3_DIR, H3_PARAMS, VESSEL_TYPE, get_data_paths,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_h3_graph(h3_df):
    """Nodes = visited (emission) cells only; edges = grid_ring neighbors."""
    h3_active = h3_df[h3_df["emission_tco2e"] > 0].copy()
    h3_ids = h3_active["h3_id"].values
    emissions = h3_active.set_index("h3_id")["emission_tco2e"].to_dict()
    h3_to_idx = {h: i for i, h in enumerate(h3_ids)}
    n = len(h3_ids)
    rows, cols, weights = [], [], []
    for i, h in enumerate(h3_ids):
        try:
            neighbors = h3.grid_ring(h, 1)
        except Exception:
            continue
        for nb in neighbors:
            j = h3_to_idx.get(nb)
            if j is not None:
                w = 1.0 / (np.log1p(emissions.get(nb, 0.0)) + 0.01)
                rows.append(i)
                cols.append(j)
                weights.append(w)
    graph = sparse.csr_matrix((weights, (rows, cols)), shape=(n, n))
    logger.info(f"    Graph: {n:,} nodes (AIS-visited cells), {graph.nnz:,} edges")
    return graph, h3_to_idx, h3_ids, emissions


def dijkstra_path_from_pred(pred, h3_ids, src_idx, dst_idx):
    """Extract node-id path from a predecessors array (single source)."""
    path_indices = []
    idx = dst_idx
    while idx != src_idx:
        path_indices.append(idx)
        idx = pred[idx]
        if idx == -9999 or idx < 0:
            return None
    path_indices.append(src_idx)
    path_indices.reverse()
    return [h3_ids[i] for i in path_indices]


class NearestH3Cache:
    """Cached nearest in-graph H3 cell for port coordinates."""

    def __init__(self, h3_to_idx, res=5):
        self.h3_to_idx = h3_to_idx
        self.res = res
        self.cache = {}

    def find(self, lat, lon):
        key = (round(lat, 4), round(lon, 4))
        if key in self.cache:
            return self.cache[key]
        cell = h3.latlng_to_cell(lat, lon, self.res)
        if cell in self.h3_to_idx:
            self.cache[key] = cell
            return cell
        for k in range(1, 30):
            for c in h3.grid_disk(cell, k):
                if c in self.h3_to_idx:
                    self.cache[key] = c
                    return c
        self.cache[key] = None
        return None


def build_corridor_h3_paths(vessel_type: str):
    """All-corridor Dijkstra on the AIS-cell-only graph (skip, never GC)."""
    paths = get_data_paths(vessel_type)
    logger.info(f"Building H3 corridor paths v4 (AIS-cell-only graph): {paths['label_cn']}")

    corridors = pd.read_parquet(PROCESSED_DIR / f"corridors_{vessel_type}.parquet")
    ports = pd.read_parquet(PROCESSED_DIR / "ports_global.parquet")
    h3_file = H3_DIR / f"h3_emission_grid_{vessel_type}.parquet"
    if not h3_file.exists():
        raise RuntimeError(f"H3 grid not found at {h3_file}")
    h3_df = pd.read_parquet(h3_file)
    logger.info(f"  Loaded H3 grid: {len(h3_df):,} cells")

    graph, h3_to_idx, h3_ids, emissions = build_h3_graph(h3_df)
    nearest = NearestH3Cache(h3_to_idx, res=H3_PARAMS["res_ocean"])

    ports_dedup = ports.drop_duplicates("port_code")
    port_coords = ports_dedup.set_index("port_code")[["port_lon", "port_lat"]].to_dict("index")
    # Backfill endpoint coordinates from the corridor table itself.
    n_backfill = 0
    for side in ("origin", "dest"):
        for _, r in corridors.iterrows():
            code = r[f"{side}_port"]
            if code not in port_coords and not pd.isna(r[f"{side}_lon"]):
                port_coords[code] = {"port_lon": r[f"{side}_lon"],
                                     "port_lat": r[f"{side}_lat"]}
                n_backfill += 1
    if n_backfill:
        logger.info(f"  Backfilled {n_backfill} endpoint coords from corridor table")

    n_total = len(corridors)
    n_dijkstra = n_fail = n_skip = 0
    n_origin = 0
    all_paths = []
    t0 = time.time()

    for o, grp in corridors.groupby("origin_port"):
        n_origin += 1
        o_c = port_coords.get(o)
        if o_c is None or pd.isna(o_c["port_lon"]):
            n_skip += len(grp)
            continue
        src_h3 = nearest.find(o_c["port_lat"], o_c["port_lon"])
        if src_h3 is None:
            # endpoint never visited by AIS → treated as land → skip
            n_skip += len(grp)
            continue
        src_idx = h3_to_idx[src_h3]
        dist, pred = dijkstra(graph, directed=False, indices=src_idx,
                              return_predecessors=True)
        for _, row in grp.iterrows():
            d_c = port_coords.get(row["dest_port"])
            if d_c is None or pd.isna(d_c["port_lon"]):
                n_skip += 1
                continue
            dst_h3 = nearest.find(d_c["port_lat"], d_c["port_lon"])
            dst_idx = h3_to_idx.get(dst_h3) if dst_h3 is not None else None
            cells = None
            if dst_idx is not None and not np.isinf(dist[dst_idx]):
                cells = dijkstra_path_from_pred(pred, h3_ids, src_idx, dst_idx)
                if cells is not None:
                    n_dijkstra += 1
                else:
                    n_fail += 1
            else:
                n_fail += 1
            if cells is None:
                # unreachable on the AIS-cell network → skip (no line drawn)
                n_skip += 1
                continue
            for seq, cell in enumerate(cells):
                all_paths.append({"corridor_id": row["corridor_id"],
                                  "h3_id": cell, "seq_order": seq,
                                  "emission_tco2e": emissions.get(cell, 0.0)})

        if n_origin % 50 == 0:
            logger.info(f"  ... {n_origin} origins / {len(all_paths):,} rows "
                        f"(Dijkstra {n_dijkstra:,} / fail {n_fail:,} / "
                        f"skip {n_skip:,}, {time.time()-t0:.0f}s)")

    logger.info(f"  RESULT: Dijkstra {n_dijkstra:,} / fail {n_fail:,} / "
                f"skip {n_skip:,} over {n_total:,} corridors ({time.time()-t0:.0f}s)")

    paths_df = pd.DataFrame(all_paths)
    coverage = paths_df.groupby("h3_id").agg(
        n_corridors=("corridor_id", "nunique"),
        total_corridor_emission=("emission_tco2e", "sum"),
    ).reset_index()
    coverage["lat"] = coverage["h3_id"].apply(lambda h: h3.cell_to_latlng(h)[0])
    coverage["lon"] = coverage["h3_id"].apply(lambda h: h3.cell_to_latlng(h)[1])
    logger.info(f"  Path entries: {len(paths_df):,}; covered cells: {len(coverage):,}; "
                f"max corridors/cell: {coverage['n_corridors'].max()}")

    paths_df.to_parquet(H3_DIR / f"corridor_h3_paths_{vessel_type}.parquet", index=False)
    coverage.to_parquet(H3_DIR / f"h3_corridor_coverage_{vessel_type}.parquet", index=False)
    logger.info("  Saved: corridor_h3_paths / h3_corridor_coverage (v4)")
    return paths_df, coverage


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vessel-type", default=VESSEL_TYPE)
    args = parser.parse_args()
    build_corridor_h3_paths(args.vessel_type)
