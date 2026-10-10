"""Independent validation gates for the R3.4 SINR-FP allocator."""
from __future__ import annotations
from charm.paths import run_root, data_directory

import json
import os
import random
import sys
from pathlib import Path

import numpy as np


ROOT = run_root()
HERE = ROOT / "phy"
DATA = data_directory()
OUT = HERE / "results_r3_4"
os.environ.setdefault("CHARM_PHY_LEGACY", "1")

import charm.physical_layer as base  # noqa: E402
from charm.power_allocation import (  # noqa: E402
    brute_force_power_grid,
    build_fixed_beamformer_groups,
    evaluate_fixed_beamformers,
    solve_sinr_fp_multistart,
    uncapped_sum_se,
)


def feasible_associations(count: int) -> list[tuple[int, ...]]:
    rng = random.Random(3401)
    selected = []
    while len(selected) < count:
        value = tuple(rng.randrange(5) for _ in range(8))
        loads = np.bincount(value, minlength=5)
        if loads.max() <= 3 and len(set(value)) >= 3 and value not in selected:
            selected.append(value)
    return selected


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scn = base.Scenario(str(DATA))
    associations = [(0, 0, 0, 1, 1, 1, 3, 3)] + feasible_associations(11)
    group_seen = set()
    grid_records = []
    dispatcher_max_rate_error = 0.0
    dispatcher_max_sinr_error = 0.0
    dispatcher_sinr_allclose = True
    for association in associations:
        paired = evaluate_fixed_beamformers(scn, association)
        direct_sinrs, direct_rates = scn._dispatch_impl(np.asarray(association), 1)
        dispatcher_max_rate_error = max(
            dispatcher_max_rate_error,
            float(np.max(np.abs(np.asarray(direct_rates) - paired["baseline_rates_gbps"]))),
        )
        dispatcher_max_sinr_error = max(
            dispatcher_max_sinr_error,
            float(np.max(np.abs(np.asarray(direct_sinrs) - paired["baseline_sinrs"]))),
        )
        dispatcher_sinr_allclose = dispatcher_sinr_allclose and bool(np.allclose(
            direct_sinrs, paired["baseline_sinrs"], rtol=1e-9, atol=1e-11
        ))
        groups, _ = build_fixed_beamformer_groups(scn, association)
        for group in groups:
            key = (group.cu_slot, group.members)
            if key in group_seen or len(grid_records) >= 20:
                continue
            group_seen.add(key)
            fp_power, fp_diag = solve_sinr_fp_multistart(group.gain, base.BW * base.N0_LIN)
            grid_power, grid_objective = brute_force_power_grid(
                group.gain, base.BW * base.N0_LIN, resolution=1000
            )
            fp_objective = uncapped_sum_se(group.gain, fp_power, base.BW * base.N0_LIN)
            relative_gap = max(0.0, grid_objective - fp_objective) / max(abs(grid_objective), 1e-15)
            grid_records.append({
                "association": list(association),
                "cu_slot": group.cu_slot,
                "members": list(group.members),
                "K": len(group.members),
                "fp_power_w": fp_power.tolist(),
                "grid_power_w": grid_power.tolist(),
                "fp_objective_sum_se": fp_objective,
                "grid_objective_sum_se": grid_objective,
                "relative_grid_upper_gap": relative_gap,
                "fp_iterations": fp_diag["iterations"],
                "fp_converged": fp_diag["converged"],
            })
    max_grid_gap = max(record["relative_grid_upper_gap"] for record in grid_records)
    payload = {
        "status": "passed" if (
            dispatcher_max_rate_error <= 1e-11
            and dispatcher_sinr_allclose
            and max_grid_gap <= 1e-3
        ) else "failed",
        "associations_checked_against_dispatcher": len(associations),
        "groups_checked_against_dense_grid": len(grid_records),
        "dispatcher_max_abs_rate_error_gbps": dispatcher_max_rate_error,
        "dispatcher_max_abs_sinr_error": dispatcher_max_sinr_error,
        "max_relative_grid_upper_gap": max_grid_gap,
        "power_budget_w": base.P_TX_LIN,
        "grid_resolution": 1000,
        "records": grid_records,
    }
    destination = OUT / "solver_validation.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))
    print(f"Saved {destination}")
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
