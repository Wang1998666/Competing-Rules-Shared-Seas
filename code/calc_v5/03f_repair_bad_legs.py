# -*- coding: utf-8 -*-
"""
Module 03f (repair, 2026-08-24): 脏航段 GC 伪影手术式修补
==========================================================
背景：3 条损坏航段（端点坐标为极地虚假值，如 dest_lat=-59.6 南极）
在 03 归因中经大圆回退在南大洋等无航线区域画出直线伪影。
03_h3_attribution 已增加 _leg_endpoints_plausible 端点校验。

本脚本对已产出的 h3_grid_v5 做手术式修补（不全量重跑归因）：
1. 从 parts_container 中移除问题船舶（BAD_MMSI）的全部旧行；
2. 用修复后的 03 process_vessel 重新计算这些船的贡献并写入新 part；
3. 重新聚合 parts → h3_emission_grid_int / h3_emission_grid / h3_stats。
修补后需依次重跑 03e / 03b / 03c / 12。
"""
import sys
import json
import logging
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import h3.api.basic_int as h3i

sys.path.insert(0, str(Path(__file__).parent))
from config import H3_DIR, PROCESSED_DIR, VESSEL_TYPE, get_data_paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("03f_repair")

BAD_MMSI = [636024385, 413900966, 413902148]


def repair(vessel_type: str):
    m03 = importlib.import_module("03_h3_attribution")
    PART_DIR = H3_DIR / f"parts_{vessel_type}"
    part_files = sorted(PART_DIR.glob("part_*.parquet"))
    logger.info(f"Parts: {len(part_files)} files; removing MMSI {BAD_MMSI}")

    # ---- 1. 移除问题船的旧行 ----
    n_removed = 0
    em_removed = 0.0
    for pf in part_files:
        df = pd.read_parquet(pf)
        keep = ~df["mmsi"].isin(BAD_MMSI)
        if (~keep).any():
            n_removed += int((~keep).sum())
            em_removed += float(df.loc[~keep, "emission_tco2e"].sum())
            df.loc[keep].to_parquet(pf, index=False)
    logger.info(f"  Removed {n_removed:,} rows ({em_removed/1e3:.2f} kt) from parts")

    # ---- 2. 用修复后的代码重算这些船 ----
    paths = get_data_paths(vessel_type)
    emissions = pd.read_parquet(PROCESSED_DIR / f"emissions_{vessel_type}.parquet")
    emissions = emissions[emissions["emission_tco2e"] > 0]
    traj_map = {}
    for traj_dir in paths["traj_dirs"]:
        if traj_dir.exists():
            for f in traj_dir.glob("*.csv"):
                traj_map[int(f.stem)] = str(f)

    recs = []
    for m in BAD_MMSI:
        sub = emissions[emissions["mmsi"] == m]
        if not len(sub):
            logger.info(f"  MMSI {m}: no emission legs, skipped")
            continue
        r = m03.process_vessel(m, traj_map.get(m), sub)
        if len(r):
            r = r.copy()
            r["mmsi"] = m
            recs.append(r)
            logger.info(f"  MMSI {m}: {len(r):,} cell rows, "
                        f"{r['emission_tco2e'].sum()/1e3:.2f} kt re-attributed "
                        f"(leg total {sub['emission_tco2e'].sum()/1e3:.2f} kt)")
        else:
            logger.info(f"  MMSI {m}: no cells after repair "
                        f"({sub['emission_tco2e'].sum()/1e3:.2f} kt dropped as garbage)")
    if recs:
        new_part = pd.concat(recs, ignore_index=True)
        out_f = PART_DIR / f"part_{len(part_files):04d}.parquet"
        new_part.to_parquet(out_f, index=False)
        logger.info(f"  Saved repaired part: {out_f.name}")

    # ---- 3. 重新聚合 ----
    logger.info("Re-aggregating all parts ...")
    part_files = sorted(PART_DIR.glob("part_*.parquet"))
    allr = pd.concat([pd.read_parquet(p) for p in part_files], ignore_index=True)
    agg = allr.groupby("h3_int")["emission_tco2e"].sum().reset_index()
    agg.to_parquet(H3_DIR / f"h3_emission_grid_int_{vessel_type}.parquet", index=False)
    agg["h3_id"] = agg["h3_int"].apply(lambda c: h3i.int_to_str(int(c)))
    agg["lat"] = agg["h3_int"].apply(lambda c: h3i.cell_to_latlng(int(c))[0])
    agg["lon"] = agg["h3_int"].apply(lambda c: h3i.cell_to_latlng(int(c))[1])
    agg["resolution"] = agg["h3_int"].apply(lambda c: h3i.get_resolution(int(c)))
    agg = agg.sort_values("emission_tco2e", ascending=False).reset_index(drop=True)

    total_h3 = agg["emission_tco2e"].sum()
    total_leg = emissions["emission_tco2e"].sum()
    logger.info(f"  H3 cells: {len(agg):,} | total {total_h3/1e6:.4f} Mt "
                f"| conservation {total_h3/total_leg:.6%} "
                f"(garbage legs dropped: {total_leg - total_h3:.1f} t)")

    agg[["h3_id", "lat", "lon", "resolution", "emission_tco2e"]].to_parquet(
        H3_DIR / f"h3_emission_grid_{vessel_type}.parquet", index=False)
    p75 = float(agg["emission_tco2e"].quantile(0.75))
    stats = {"version": "v2-trajectory", "n_cells": int(len(agg)),
             "total_Mt": round(total_h3 / 1e6, 2), "p75_t": p75,
             "conservation_ratio": float(total_h3 / total_leg),
             "n_res5": int((agg["resolution"] == 5).sum()),
             "n_res7": int((agg["resolution"] == 7).sum())}
    with open(H3_DIR / f"h3_stats_{vessel_type}.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("  Saved grid + stats. DONE. 请继续重跑 03e → 03b → 03c → 12")


if __name__ == "__main__":
    repair(VESSEL_TYPE)
