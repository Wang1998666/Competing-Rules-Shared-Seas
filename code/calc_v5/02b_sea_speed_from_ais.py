"""
Module 02b (v5): Trajectory-Resolved Load Factor (correction C4)
=================================================================
Implements factor (4) of the emission-gap diagnosis with the physically
correct time integral of the cubic load law, replacing the leg-average
cube law diluted by slow segments (Jensen gap: mean(v^3) >= mean(v)^3):

    E_odv = P_ME(75%) * [ ∫_leg clip[(SOG(t)/v_design)^3, 0.2, 1] dt ] * EF_WtW

Per leg, the integral runs over the vessel's UNDERWAY (sailing=True) AIS
points whose timestamps fall inside the leg window [leg_start_time,
leg_end_time]. The provider's leg windows tile the year contiguously
(arrival at destination -> next departure), so every underway point is
attributed to exactly one leg and port stays are excluded by the sailing
flag. AIS gaps are bridged with the preceding point's load factor when
shorter than DT_CAP_H (6 h) and ignored beyond that (blackout).

Legs whose vessel has no trajectory file, or whose window contains fewer
than 2 underway points, get NaN and fall back to the leg-average form in
module 02.

Verification (calc_v5/_verify_factor4_result.json, 98-vessel stratified
sample): aggregate ratio 1.17, per-vessel median 1.24 vs the leg-average
formula -> adopted per gap-report decision D1 (option c, partial zone).

Output: 02_数据_output/processed_v5/traj_leg_integrals_{vt}.parquet
        columns mmsi, origin_port, dest_port, leg_start_time,
        traj_integ_h, traj_t_under_h, traj_med_sog_kn, v_design_used
"""

import sys
import argparse
import logging
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PROCESSED_DIR, EMISSION_PARAMS, PROCESSING_PARAMS, VESSEL_TYPE,
    get_data_paths,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DT_CAP_H = 6.0          # AIS gap bridge cap (hours); longer gaps ignored
MIN_SOG_KN = 0.0        # valid SOG range
MAX_SOG_KN = 40.0


def build_traj_map(paths) -> dict:
    """MMSI -> trajectory CSV path over all provider trajectory dirs."""
    traj_map = {}
    for d in paths["traj_dirs"]:
        if d.exists():
            for f in d.glob("*.csv"):
                try:
                    traj_map[int(f.stem)] = str(f)
                except ValueError:
                    continue
    return traj_map


def process_vessel(args):
    """Per-vessel per-leg underway LF integral hours. Returns a list of
    row tuples or None when the trajectory is unusable."""
    mmsi, traj_file, leg_arr, v_design = args
    # leg_arr: (starts_ns, ends_ns, origins, dests) as numpy arrays
    try:
        df = pd.read_csv(traj_file, usecols=["postime", "sog", "sailing"],
                         dtype={"postime": str, "sailing": str})
    except Exception:
        return None
    df = df.dropna(subset=["sog", "postime"])
    if len(df) < 10:
        return None
    sog = pd.to_numeric(df["sog"], errors="coerce")
    ok = (sog >= MIN_SOG_KN) & (sog <= MAX_SOG_KN)
    ok &= df["sailing"].astype(str).str.lower().values == "true"
    df = df[ok]
    if len(df) < 10:
        return None
    t = pd.to_datetime(df["postime"], errors="coerce")
    keep = t.notna()
    df, t, sog = df[keep], t[keep], sog[ok][keep]
    if len(df) < 10:
        return None
    order = np.argsort(t.values)
    t_sec = t.values.astype("datetime64[s]").astype(np.int64)[order]
    sog_arr = sog.values[order]

    lf_min = EMISSION_PARAMS["load_factor_min"]
    lf_max = EMISSION_PARAMS["load_factor_max"]
    with np.errstate(divide="ignore", invalid="ignore"):
        lf = np.clip((sog_arr / v_design) ** 3, lf_min, lf_max)

    dt_cap_s = DT_CAP_H * 3600.0
    rows = []
    starts_ns, ends_ns, origins, dests = leg_arr
    for i in range(len(starts_ns)):
        t0 = int(starts_ns[i]) // 1_000_000_000      # ns -> s
        t1 = int(ends_ns[i]) // 1_000_000_000
        if t1 <= t0:
            continue
        a = np.searchsorted(t_sec, t0, side="left")
        b = np.searchsorted(t_sec, t1, side="right")
        if b - a < 2:
            rows.append((mmsi, origins[i], dests[i], starts_ns[i],
                         np.nan, np.nan, np.nan, v_design))
            continue
        dt = np.clip(np.diff(t_sec[a:b]), 0, dt_cap_s) / 3600.0
        integ_h = float((lf[a:b - 1] * dt).sum())
        t_under = float(dt.sum())
        med_sog = float(np.median(sog_arr[a:b]))
        rows.append((mmsi, origins[i], dests[i], starts_ns[i],
                     integ_h, t_under, med_sog, v_design))
    return rows


def run(vessel_type: str):
    paths = get_data_paths(vessel_type)
    logger.info(f"Module 02b (v5): trajectory-resolved load factor — {paths['label_cn']}")

    legs = pd.read_parquet(PROCESSED_DIR / f"legs_{vessel_type}.parquet")
    vessels = pd.read_parquet(PROCESSED_DIR / f"vessels_{vessel_type}.parquet")
    traj_map = build_traj_map(paths)
    logger.info(f"  Legs: {len(legs):,}; vessels: {legs['mmsi'].nunique():,}; "
                f"trajectory files: {len(traj_map):,}")

    vdesign = vessels.set_index("mmsi")["design_speed_kn"].to_dict()
    med_vd = float(np.median([v for v in vdesign.values() if v and v > 0]))

    legs = legs.dropna(subset=["leg_start_time", "leg_end_time"]).copy()
    tasks = []
    n_no_traj = 0
    for mmsi, g in legs.groupby("mmsi", sort=False):
        f = traj_map.get(int(mmsi))
        if f is None:
            n_no_traj += len(g)
            continue
        vd = vdesign.get(mmsi, np.nan)
        if not (vd and vd > 0):
            vd = med_vd
        leg_arr = (
            g["leg_start_time"].values.astype("datetime64[ns]").astype(np.int64),
            g["leg_end_time"].values.astype("datetime64[ns]").astype(np.int64),
            g["origin_port"].values,
            g["dest_port"].values,
        )
        tasks.append((int(mmsi), f, leg_arr, float(vd)))
    logger.info(f"  Vessels with trajectory: {len(tasks):,} "
                f"(legs without trajectory file: {n_no_traj:,})")

    t0 = time.time()
    all_rows = []
    n_workers = PROCESSING_PARAMS["n_workers"]
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(process_vessel, t) for t in tasks]
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                all_rows.extend(r)
            done += 1
            if done % 500 == 0:
                logger.info(f"  {done:,}/{len(tasks):,} vessels "
                            f"({time.time() - t0:.0f}s)")

    cols = ["mmsi", "origin_port", "dest_port", "leg_start_time",
            "traj_integ_h", "traj_t_under_h", "traj_med_sog_kn", "v_design_used"]
    res = pd.DataFrame(all_rows, columns=cols)
    res["leg_start_time"] = pd.to_datetime(res["leg_start_time"])
    res = res.drop_duplicates(
        subset=["mmsi", "origin_port", "dest_port", "leg_start_time"], keep="first")

    n_matched = int(res["traj_integ_h"].notna().sum())
    logger.info(f"\n  02b RESULTS (v5):")
    logger.info(f"    Legs with window integral: {n_matched:,} / {len(res):,} "
                f"({n_matched / max(len(res), 1) * 100:.1f}%)")
    logger.info(f"    Total underway hours: {res['traj_t_under_h'].sum():,.0f}")
    logger.info(f"    Total LF-integral hours: {res['traj_integ_h'].sum():,.0f}")
    logger.info(f"    Mean underway LF (integral/underway): "
                f"{(res['traj_integ_h'].sum() / max(res['traj_t_under_h'].sum(), 1e-9)):.3f}")

    out = PROCESSED_DIR / f"traj_leg_integrals_{vessel_type}.parquet"
    res.to_parquet(out, index=False)
    logger.info(f"  Saved: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    logger.info(f"  Elapsed: {(time.time() - t0) / 60:.1f} min")
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trajectory LF integrals (v5 C4)")
    parser.add_argument("--vessel-type", default=VESSEL_TYPE,
                        choices=["container", "tanker"])
    args = parser.parse_args()
    run(args.vessel_type)
