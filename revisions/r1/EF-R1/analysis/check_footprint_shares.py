"""Reproduce Table 3 and the narrower BRI footprint in Supporting Information Text S2."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ANALYSIS = Path(__file__).resolve().parent
DATA = ANALYSIS / "inputs" / "data"
GROUPS = ("EU-connected", "BRI-only", "Near-EU", "Outside footprints")
ARCHIVED_LABELS = {
    "EU_ETS": GROUPS[0],
    "BRI_only": GROUPS[1],
    "FuelEU_exposed": GROUPS[2],
    "Vacuum": GROUPS[3],
}


def footprint_shares():
    policy = DATA / "derived" / "policy_analysis_v5"
    ports = pd.read_parquet(
        DATA / "derived" / "processed_v5" / "candidate_ports_container.parquet",
        columns=["port_code", "country_code"],
    ).drop_duplicates("port_code")
    routes = pd.read_parquet(policy / "governance_corridors_container.parquet")
    codes = set(pd.read_csv(DATA / "institutional_lists" / "bri_maritime_partners.csv")["iso2"])
    exclusions = {"IT", "TW", "IN"}
    if len(codes) != 59 or not exclusions.issubset(codes):
        raise ValueError("Unexpected reference footprint in bri_maritime_partners.csv")
    if not routes["corridor_id"].is_unique:
        raise ValueError("The governance table must contain one row per corridor")

    country_by_port = ports.set_index("port_code")["country_code"]
    origin = routes["origin_port"].map(country_by_port)
    destination = routes["dest_port"].map(country_by_port)
    total_emissions = routes["E_e"].sum()
    results = []
    for name, members in (("reference_59", codes), ("narrow_56", codes - exclusions)):
        # An endpoint without a recorded country code receives no BRI membership flag.
        bri_link = origin.isin(members) | destination.isin(members)
        group = pd.Series(
            np.select(
                [routes["eu_ets_endpoint"], bri_link, routes["fueleu_exposed"]],
                GROUPS[:3],
                default=GROUPS[3],
            ),
            index=routes.index,
        )
        if name == "reference_59":
            if not bri_link.eq(routes["is_bri"]).all():
                raise ValueError("Reference BRI flags differ from the archived route table")
            if not group.eq(routes["gov_type"].map(ARCHIVED_LABELS)).all():
                raise ValueError("Reference groups differ from the archived classification")
        for label in GROUPS:
            selected = group.eq(label)
            emissions = routes.loc[selected, "E_e"].sum()
            results.append({
                "scenario": name,
                "group": label,
                "corridors": int(selected.sum()),
                "annual_emissions_Mt_CO2e": emissions / 1_000_000,
                "share_of_corridor_emissions_pct": 100 * emissions / total_emissions,
            })
    return pd.DataFrame(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write the summary to this CSV (default: stdout)")
    args = parser.parse_args()
    footprint_shares().to_csv(
        args.output or sys.stdout, index=False, float_format="%.8f", lineterminator="\n"
    )
