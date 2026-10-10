"""Validation gates for Reviewer-3.1 iterative SLAPP."""
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
OUT = HERE / "results_r3_1"
os.environ.setdefault("CHARM_PHY_LEGACY", "1")

from charm.physical_layer import Scenario  # noqa: E402
from charm.power_allocation import evaluate_iterative_slapp  # noqa: E402


def associations(count: int) -> list[tuple[int, ...]]:
    rng = random.Random(3101)
    values = [(0, 0, 0, 1, 1, 1, 3, 3)]
    while len(values) < count:
        candidate = tuple(rng.randrange(5) for _ in range(8))
        loads = np.bincount(candidate, minlength=5)
        if loads.max() <= 3 and len(set(candidate)) >= 3 and candidate not in values:
            values.append(candidate)
    return values


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scn = Scenario(str(DATA))
    group_cache = {}
    records = []
    max_rate_error = 0.0
    for association in associations(30):
        result = evaluate_iterative_slapp(scn, association, group_cache)
        direct_sinrs, direct_rates = scn._dispatch_impl(np.asarray(association), 1)
        max_rate_error = max(max_rate_error, float(np.max(
            np.abs(np.asarray(direct_rates) - result["baseline_rates_gbps"])
        )))
        for group in result["groups"]:
            trace = group["trace"]
            records.append({
                "association": list(association),
                "cu_slot": group["cu_slot"],
                "members": group["members"],
                "one_shot_sum_rate_gbps": trace[0]["sum_rate_gbps"],
                "final_sum_rate_gbps": trace[-1]["sum_rate_gbps"],
                "one_shot_min_rate_gbps": trace[0]["min_rate_gbps"],
                "final_min_rate_gbps": trace[-1]["min_rate_gbps"],
                "iterations_after_one_shot": group["iterations_after_one_shot"],
                "converged": group["converged"],
                "sum_rate_monotonic": group["sum_rate_monotonic"],
                "one_shot_power_cv": trace[0]["coefficient_of_variation"],
                "one_shot_power_max_min_ratio": trace[0]["max_min_positive_ratio"],
                "trace": trace,
            })
    payload = {
        "status": "passed" if max_rate_error <= 1e-11 else "failed",
        "full_associations_checked": 30,
        "unique_cu_du_groups": len(group_cache),
        "max_abs_iteration0_vs_dispatcher_rate_error_gbps": max_rate_error,
        "group_convergence_fraction": float(np.mean([row["converged"] for row in records])),
        "group_monotonic_fraction": float(np.mean([row["sum_rate_monotonic"] for row in records])),
        "mean_iterations_after_one_shot": float(np.mean([row["iterations_after_one_shot"] for row in records])),
        "mean_one_shot_power_cv": float(np.mean([row["one_shot_power_cv"] for row in records])),
        "median_one_shot_power_max_min_ratio": float(np.median([row["one_shot_power_max_min_ratio"] for row in records])),
        "records": records,
    }
    destination = OUT / "iterative_validation.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))
    print(f"Saved {destination}")
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
