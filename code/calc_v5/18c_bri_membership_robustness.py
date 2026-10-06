"""
Module 18c (v4): BRI Membership Robustness — Italy Exit & List Currency
=======================================================================
Addresses review point D.1: Italy formally withdrew from the Belt and Road
Initiative in December 2023 (effective before the 2024 study year), yet the
59-country maritime list used by the classifier still contains Italy; the
list also contains Taiwan, which is not a BRI signatory partner.

Because Italy is an EU/EEA member, corridors with an Italian endpoint are
already classified EU_ETS under the priority convention, so the mutually
exclusive governance shares are unaffected; what changes is the non-exclusive
BRI membership flag (BRI emission share, BRI∩ETS overlap, the M^BRI
investment domain). Taiwan is not an EEA member, so TW-only-BRI corridors
would migrate between non-EU categories.

This module recomputes the governance statistics under three list variants:
  * reference (59 countries incl. IT, TW)
  * ex-Italy  (58)
  * ex-Italy-Taiwan (57)

Outputs (02_数据_output/policy_analysis_v4/):
  bri_membership_robustness_container.json
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import PROCESSED_DIR, POLICY_DIR, VESSEL_TYPE
from importlib import import_module
_08 = import_module("08_bri_analysis")
BRI_MARITIME_COUNTRIES = _08.BRI_MARITIME_COUNTRIES
_03c = import_module("03c_h3_policy_coverage")
EU_EEA_COUNTRIES = _03c.EU_EEA_COUNTRIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def recompute(gov, bri_set):
    o_bri = gov["o_cc"].isin(bri_set)
    d_bri = gov["d_cc"].isin(bri_set)
    is_bri = o_bri | d_bri
    tot = gov["E_e"].sum()

    cond_eu = gov["eu_ets_endpoint"]
    fuel = gov["fueleu_exposed"]
    gt = np.select([cond_eu, ~cond_eu & is_bri, ~cond_eu & ~is_bri & fuel],
                   ["EU_ETS", "BRI_only", "FuelEU_exposed"], default="Vacuum")
    gt_old = gov["gov_type"]

    vac = gt == "Vacuum"
    p75 = gov.loc[gov["E_e"] > 0, "E_e"].quantile(0.75)
    vac_strict = vac & (gov["E_e"] >= p75)

    overlap = int((is_bri & cond_eu).sum())
    res = {
        "bri_emission_share_pct": float(gov.loc[is_bri, "E_e"].sum() / tot * 100),
        "n_bri_corridors": int(is_bri.sum()),
        "n_bri_ets_overlap": overlap,
        "gov_shares_pct": {k: float(gov.loc[gt == k, "E_e"].sum() / tot * 100)
                           for k in ["EU_ETS", "BRI_only", "FuelEU_exposed", "Vacuum"]},
        "rvi_any_pct": float(gov.loc[vac, "E_e"].sum() / tot * 100),
        "rvi_strict_pct": float(gov.loc[vac_strict, "E_e"].sum() / tot * 100),
        "n_vacuum_highE": int(vac_strict.sum()),
        "n_type_changed_corridors": int((gt != gt_old).sum()),
        "emission_of_changed_pct": float(gov.loc[gt != gt_old, "E_e"].sum() / tot * 100),
    }
    # BRI-camp port counts (camp = EU priority over BRI)
    return res


def main(vessel_type: str):
    gov = pd.read_parquet(POLICY_DIR / f"governance_corridors_{vessel_type}.parquet")
    ports = pd.read_parquet(PROCESSED_DIR / "ports_global.parquet").drop_duplicates("port_code")
    cc = ports.set_index("port_code")["country_code"]
    gov["o_cc"] = gov["origin_port"].map(cc)
    gov["d_cc"] = gov["dest_port"].map(cc)

    it_ports = sorted(ports.loc[ports["country_code"] == "IT", "port_name_cn"].tolist())
    tw_ports = sorted(ports.loc[ports["country_code"] == "TW", "port_name_cn"].tolist())
    logger.info(f"IT ports in data: {it_ports}")
    logger.info(f"TW ports in data: {tw_ports}")

    variants = {
        "reference": BRI_MARITIME_COUNTRIES,
        "ex_Italy": BRI_MARITIME_COUNTRIES - {"IT"},
        "ex_Italy_TW": BRI_MARITIME_COUNTRIES - {"IT", "TW"},
    }
    out = {}
    for name, bset in variants.items():
        out[name] = recompute(gov, bset)
        logger.info(f"[{name}] BRI em share {out[name]['bri_emission_share_pct']:.1f}%, "
                    f"overlap {out[name]['n_bri_ets_overlap']}, "
                    f"RVI {out[name]['rvi_strict_pct']:.1f}%, "
                    f"type-changed corridors {out[name]['n_type_changed_corridors']}")

    # which corridors actually change under ex-Italy-TW
    bri_ref = gov["o_cc"].isin(BRI_MARITIME_COUNTRIES) | gov["d_cc"].isin(BRI_MARITIME_COUNTRIES)
    bri_new = (gov["o_cc"].isin(variants["ex_Italy_TW"])
               | gov["d_cc"].isin(variants["ex_Italy_TW"]))
    lost = gov[bri_ref & ~bri_new]
    out["corridors_losing_bri_flag"] = {
        "n": int(len(lost)),
        "emission_share_pct": float(lost["E_e"].sum() / gov["E_e"].sum() * 100),
        "top_examples": [
            {"origin": r["origin_port"], "dest": r["dest_port"],
             "o_country": r["o_cc"], "d_country": r["d_cc"],
             "emission_t": float(r["E_e"])}
            for _, r in lost.nlargest(8, "E_e").iterrows()],
    }

    with open(POLICY_DIR / f"bri_membership_robustness_{vessel_type}.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logger.info("Saved bri_membership_robustness.")
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vessel-type", default=VESSEL_TYPE,
                        choices=["container", "tanker"])
    main(parser.parse_args().vessel_type)
