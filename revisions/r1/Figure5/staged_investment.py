"""Phased expansion of four methanol investment plans on the 2024 network."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack


PLANS = {"Global": ("global", False), "EU": ("EU", False),
         "BRI": ("BRI", True), "BRI_open": ("BRI", False)}
SHARES = (.05, .10, .15, .20)


def solve_plan(corridors, ports, mids, feasible, bri, closure):
    index = {code: i for i, code in enumerate(ports.port_code)}
    nports = len(index)
    costs = ports.cost_p.to_numpy()
    rows, cols, vals = [], [], []
    limits = []

    def constraint(coefficients, limit=0):
        for col, val in coefficients.items():
            rows.append(len(limits))
            cols.append(col)
            vals.append(val)
        limits.append(limit)

    for j, route in enumerate(corridors.itertuples()):
        active = nports + j
        endpoints = {route.origin_port, route.dest_port}
        if not endpoints <= feasible:
            constraint({active: 1})
            continue
        for ep in endpoints:
            constraint({active: 1, index[ep]: -1})
        if route.mean_distance > 8000:
            required = max(1, int(route.k_e_green_methanol))
            candidates = {index[code]: -1 for code in mids.get(route.corridor_id, [])}
            constraint({active: required, **candidates})

    constraint(dict(enumerate(costs)))
    N = nports + len(corridors)
    A = coo_matrix((vals, (rows, cols)), shape=(len(limits), N)).tocsc()
    integer = np.r_[np.ones(nports), np.zeros(len(corridors))]
    W = corridors.W.to_numpy() / 1e6
    assert np.isfinite(W).all() and (W >= 0).all()
    scores = {"global": W, "EU": W * corridors.EU, "BRI": W * corridors.BRI}
    scores = {name: np.r_[np.zeros(nports), weight] for name, weight in scores.items()}
    results, selections = [], []
    for plan, (scope, closed) in PLANS.items():
        upper = np.ones(N)
        if closed:
            upper[:nports] = ports.country_code.isin(bri).to_numpy()
        already_built = set()
        previous = None
        for step, share in enumerate((*SHARES, SHARES[-1]), 1):
            fresh = step > len(SHARES)
            locked = set() if fresh else already_built
            lower = np.zeros(N)
            lower[[index[code] for code in locked]] = 1
            budget = costs.sum() * share
            limits[-1] = budget
            first = milp(-scores[scope], integrality=integer, bounds=Bounds(lower, upper),
                         constraints=LinearConstraint(A, -np.inf, limits),
                         options={"time_limit": 80, "mip_rel_gap": 1e-6})
            assert first.success and first.mip_gap <= 1.01e-6, (plan, step, first)
            fitted = first
            if scope != "global":
                combined = vstack([A, coo_matrix(scores[scope].reshape(1, -1))], format="csc")
                fitted = milp(-scores["global"], integrality=integer, bounds=Bounds(lower, upper),
                              constraints=LinearConstraint(combined,
                                                           np.r_[np.full(len(limits), -np.inf), -first.fun-1e-6],
                                                           np.r_[limits, np.inf]),
                              options={"time_limit": 80, "mip_rel_gap": 1e-5})
                assert fitted.success and fitted.mip_gap <= 1.01e-5, (plan, step, fitted)
            chosen = set(ports.loc[fitted.x[:nports] > .5, "port_code"])
            actual = corridors[closure(corridors, chosen, mids, feasible)]
            row = {"plan": plan, "mode": "fresh" if fresh else "staged",
                   "stage": min(step,len(SHARES)), "budget_pct": round(share*100),
                   "budget_MUSD": budget, "cost_MUSD": float(costs[fitted.x[:nports] > .5].sum()),
                   "n_ports": len(chosen), "n_new_ports": len(chosen-locked),
                   "positive_routes": int((actual.W>0).sum()), "global_Mt": actual.W.sum()/1e6,
                   "EU_Mt": actual.loc[actual.EU,"W"].sum()/1e6, "BRI_Mt": actual.loc[actual.BRI,"W"].sum()/1e6,
                   "primary_gap": first.mip_gap, "final_gap": fitted.mip_gap,
                   "primary_bound_Mt": -first.mip_dual_bound}
            assert locked <= chosen and row["cost_MUSD"] <= budget + 1e-5
            if closed:
                assert set(ports.loc[ports.port_code.isin(chosen), "country_code"]) <= bri
            assert row[f"{scope}_Mt"] >= -first.fun - 2e-6, row
            if not fresh:
                assert previous is None or all(row[f"{key}_Mt"] >= previous[f"{key}_Mt"] - 1e-8
                                               for key in scores)
                already_built = chosen
                previous = row
            results.append(row)
            selections.extend({"plan":plan,"stage":row["stage"],"mode":row["mode"],
                               "port_code":code,"added":code not in locked} for code in sorted(chosen))
            print(plan, row["mode"], row["budget_pct"], f'{row["global_Mt"]:.3f} Mt, {len(chosen)} ports', flush=True)
    return pd.DataFrame(results), pd.DataFrame(selections)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path, help="Directory with the archived analysis inputs")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.analysis.resolve()))
    from audit_activation import closure, load
    from recompute_rule_comparison import literal_set
    corridors, ports, _, governance, mids, feasible = load()
    ports = ports.loc[ports.is_feasible].reset_index(drop=True)
    governance = governance.set_index("corridor_id")
    corridors["EU"] = corridors.corridor_id.map(governance.eu_ets_endpoint).astype("boolean").fillna(False).astype(bool)
    corridors["BRI"] = corridors.corridor_id.map(governance.is_bri).astype("boolean").fillna(False).astype(bool)
    bri = literal_set("08_bri_analysis.py", "BRI_MARITIME_COUNTRIES")
    results, selections = solve_plan(corridors, ports, mids, feasible, bri, closure)
    args.output.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output/"staged_results.csv", index=False)
    selections.to_csv(args.output/"staged_ports.csv", index=False)


if __name__ == "__main__":
    main()

