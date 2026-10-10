"""Exact Tier-2 joint CU-deployment/DU-association enumeration.

This is intentionally independent of the old `classic_baslines.m` association
heuristic. It enumerates every assignment of D=8 policy positions to C=5 CU
slots subject to N_RF=3. The active deployment is the set of CU slots actually
used by the assignment, so redundant unused active-CU subsets are not counted.

SLAPP rates come from the audited Python port of the submitted MATLAB
dispatcher. Multiple feasibility frontiers are retained instead of silently
choosing between the submitted 90%-QoS reporting tolerance, the exact 6-Gbps
constraint, and the Eq.19/20 target-power feasibility issue.
"""
from __future__ import annotations
from charm.paths import run_root, data_directory

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Iterator

import numpy as np


ROOT = run_root()
DATA = data_directory()
DEFAULT_OUT = ROOT / "tier2_joint_enumeration" / "results"

os.environ.setdefault("CHARM_PHY_LEGACY", "1")
from charm.physical_layer import ICU_LOS, Scenario  # noqa: E402


C, D, NRF, PA = 5, 8, 3, 1
R_MIN = 6.0
SOFT_FACTOR = 0.9


def enumerate_capacity_feasible(c: int = C, d: int = D, capacity: int = NRF) -> Iterator[tuple[int, ...]]:
    """Yield every labelled association with per-CU load <= capacity once."""
    loads = [0] * c
    association = [0] * d

    def visit(position: int):
        if position == d:
            yield tuple(association)
            return
        for cu in range(c):
            if loads[cu] >= capacity:
                continue
            association[position] = cu
            loads[cu] += 1
            yield from visit(position + 1)
            loads[cu] -= 1

    yield from visit(0)


def analytical_counts(c: int = C, d: int = D, capacity: int = NRF) -> dict[int, int]:
    """Independent multinomial count by number of nonempty labelled CUs."""
    counts = {k: 0 for k in range(1, c + 1)}

    def positive_occupancies(k: int, remaining: int, prefix: tuple[int, ...] = ()):
        if len(prefix) == k:
            if remaining == 0:
                yield prefix
            return
        slots_left = k - len(prefix) - 1
        for value in range(1, capacity + 1):
            rest = remaining - value
            if slots_left <= rest <= slots_left * capacity:
                yield from positive_occupancies(k, rest, prefix + (value,))

    for k in range(1, c + 1):
        per_chosen_subset = 0
        for occupancy in positive_occupancies(k, d):
            ways = math.factorial(d)
            for value in occupancy:
                ways //= math.factorial(value)
            per_chosen_subset += ways
        counts[k] = math.comb(c, k) * per_chosen_subset
    return counts


def summarize_power(diagnostics: list[dict]) -> dict:
    finite = all(bool(np.all(item["finite_target_feasible"])) for item in diagnostics)
    budget = any(bool(item["budget_infeasible"]) for item in diagnostics)
    under = any(bool(np.any(item["under_target_after_normalization"])) for item in diagnostics)
    return {
        "finite_positive_slapp_denominators": finite,
        "target_power_within_budget": not budget,
        "no_target_reduction_after_normalization": not under,
        "strict_target_power_feasible": finite and not budget and not under,
    }


def dispatcher_du_rxids(scn: Scenario) -> list[int]:
    """Return the fixed physical DU identities used by the submitted dispatcher."""
    los_counts = scn.losMat.sum(axis=1)
    return (np.argsort(los_counts, kind="stable")[:D] + 1).astype(int).tolist()


def build_group_cache(scn: Scenario, du_rxids: list[int]) -> dict:
    """Evaluate every independent (CU, nonempty DU subset) group exactly once.

    This is exact for the submitted PHY because channel construction, CU
    boresight, precoding, SLAPP power allocation, and interference are all
    confined to the DUs served by one CU.  Physical-layer code is not changed.
    """
    cache: dict[tuple[int, tuple[int, ...]], dict] = {}
    total = C * sum(math.comb(D, size) for size in range(1, NRF + 1))
    started = time.perf_counter()
    completed = 0
    for cu_slot in range(C):
        physical_txid = int(ICU_LOS[cu_slot])
        for size in range(1, NRF + 1):
            for du_positions in combinations(range(D), size):
                physical_du_rxids = [du_rxids[position] for position in du_positions]
                sinrs, rates = scn.compute_rates(
                    [physical_txid], physical_du_rxids, [0] * size, PA
                )
                cache[(cu_slot, du_positions)] = {
                    "sinrs": np.asarray(sinrs, dtype=float).copy(),
                    "rates": np.asarray(rates, dtype=float).copy(),
                    "power": summarize_power(scn.last_diagnostics),
                }
                completed += 1
                if completed % 50 == 0 or completed == total:
                    elapsed = time.perf_counter() - started
                    print(f"group-cache={completed}/{total} elapsed={elapsed:.1f}s", flush=True)
    return cache


def evaluate_from_group_cache(association: tuple[int, ...], group_cache: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Compose one complete assignment from disjoint cached CU groups."""
    rates = np.zeros(D, dtype=float)
    sinrs = np.zeros(D, dtype=float)
    group_power = []
    for cu_slot in sorted(set(association)):
        du_positions = tuple(index for index, value in enumerate(association) if value == cu_slot)
        entry = group_cache[(cu_slot, du_positions)]
        rates[list(du_positions)] = entry["rates"]
        sinrs[list(du_positions)] = entry["sinrs"]
        group_power.append(entry["power"])
    power = {
        "finite_positive_slapp_denominators": all(
            item["finite_positive_slapp_denominators"] for item in group_power
        ),
        "target_power_within_budget": all(
            item["target_power_within_budget"] for item in group_power
        ),
        "no_target_reduction_after_normalization": all(
            item["no_target_reduction_after_normalization"] for item in group_power
        ),
    }
    power["strict_target_power_feasible"] = all(
        item["strict_target_power_feasible"] for item in group_power
    )
    return sinrs, rates, power


def parity_assignments(sample_count: int) -> list[tuple[int, ...]]:
    """Deterministic, varied validation sample including every active-CU count."""
    by_count: dict[int, list[tuple[int, ...]]] = {3: [], 4: [], 5: []}
    reservoir_seen = Counter()
    rng = random.Random(20260814)
    base, remainder = divmod(sample_count, len(by_count))
    targets = {
        k: base + int(index < remainder)
        for index, k in enumerate(sorted(by_count))
    }
    for association in enumerate_capacity_feasible():
        k = len(set(association))
        reservoir_seen[k] += 1
        bucket = by_count[k]
        target = targets[k]
        if len(bucket) < target:
            bucket.append(association)
        else:
            replacement = rng.randrange(reservoir_seen[k])
            if replacement < target:
                bucket[replacement] = association
    selected = by_count[3] + by_count[4] + by_count[5]
    rng.shuffle(selected)
    return selected[:sample_count]


def validate_group_cache(scn: Scenario, group_cache: dict, sample_count: int) -> dict:
    """Use the unchanged dispatcher as an oracle for cached composition."""
    checked = 0
    max_rate_error = 0.0
    max_sinr_error = 0.0
    started = time.perf_counter()
    for association in parity_assignments(sample_count):
        direct_sinrs, direct_rates = scn._dispatch_impl(np.asarray(association, dtype=int), PA)
        direct_power = summarize_power(scn.last_diagnostics)
        cached_sinrs, cached_rates, cached_power = evaluate_from_group_cache(association, group_cache)
        rate_error = float(np.max(np.abs(np.asarray(direct_rates) - cached_rates)))
        sinr_error = float(np.max(np.abs(np.asarray(direct_sinrs) - cached_sinrs)))
        max_rate_error = max(max_rate_error, rate_error)
        max_sinr_error = max(max_sinr_error, sinr_error)
        if not np.allclose(direct_rates, cached_rates, rtol=1e-11, atol=1e-11):
            raise RuntimeError(f"Cached rate parity failed for {association}; max error={rate_error}")
        if not np.allclose(direct_sinrs, cached_sinrs, rtol=1e-11, atol=1e-11):
            raise RuntimeError(f"Cached SINR parity failed for {association}; max error={sinr_error}")
        if direct_power != cached_power:
            raise RuntimeError(
                f"Cached power-diagnostic parity failed for {association}: "
                f"direct={direct_power}, cached={cached_power}"
            )
        checked += 1
    result = {
        "checked": checked,
        "max_abs_rate_error_gbps": max_rate_error,
        "max_abs_sinr_error": max_sinr_error,
        "elapsed_seconds": time.perf_counter() - started,
        "rtol": 1e-11,
        "atol": 1e-11,
        "status": "passed",
    }
    print(f"cache parity PASSED: {result}", flush=True)
    return result


def serializable_group_cache(group_cache: dict, du_rxids: list[int]) -> dict:
    records = []
    for (cu_slot, du_positions), entry in sorted(group_cache.items()):
        records.append({
            "cu_slot_0based": cu_slot,
            "physical_txid": int(ICU_LOS[cu_slot]),
            "du_positions_0based": list(du_positions),
            "physical_du_rxids": [du_rxids[position] for position in du_positions],
            "sinrs": entry["sinrs"].tolist(),
            "rates_gbps": entry["rates"].tolist(),
            "power": entry["power"],
        })
    return {
        "definition": "exact independent per-CU PHY results for every nonempty DU subset of size <= N_RF",
        "entry_count": len(records),
        "dispatcher_selected_du_rxids": du_rxids,
        "records": records,
    }


def candidate(association: tuple[int, ...], rates: np.ndarray, power: dict) -> dict:
    active = sorted(set(association))
    return {
        "association_cu_slots_0based": list(association),
        "active_cu_slots_0based": active,
        "active_physical_txids": [int(ICU_LOS[index]) for index in active],
        "cu_loads": np.bincount(association, minlength=C).astype(int).tolist(),
        "n_active_cus": len(active),
        "rates_gbps": [float(value) for value in rates],
        "sum_rate_gbps": float(np.sum(rates)),
        "min_rate_gbps": float(np.min(rates)),
        "exact_qos_feasible": bool(np.all(rates >= R_MIN)),
        "soft_qos_feasible": bool(np.all(rates >= SOFT_FACTOR * R_MIN)),
        "power": power,
    }


def is_better(new: dict, old: dict | None, minimum_cus_first: bool = False) -> bool:
    if old is None:
        return True
    if minimum_cus_first and new["n_active_cus"] != old["n_active_cus"]:
        return new["n_active_cus"] < old["n_active_cus"]
    if not np.isclose(new["sum_rate_gbps"], old["sum_rate_gbps"], rtol=0, atol=1e-12):
        return new["sum_rate_gbps"] > old["sum_rate_gbps"]
    return tuple(new["association_cu_slots_0based"]) < tuple(old["association_cu_slots_0based"])


def update_best(best: dict, record: dict) -> None:
    categories = ["nrf_feasible_max_sum"]
    if record["soft_qos_feasible"]:
        categories.extend(["soft_qos_max_sum", "min_cu_soft_qos_then_sum"])
    if record["exact_qos_feasible"]:
        categories.extend(["exact_qos_max_sum", "min_cu_exact_qos_then_sum"])
    if record["power"]["strict_target_power_feasible"]:
        categories.append("strict_power_max_sum")
        if record["exact_qos_feasible"]:
            categories.extend(["strict_power_exact_qos_max_sum", "min_cu_strict_power_exact_qos_then_sum"])
    for name in categories:
        minimum_first = name.startswith("min_cu_")
        if is_better(record, best.get(name), minimum_first):
            best[name] = record


def result_payload(scn: Scenario, evaluated: int, counts: Counter, flags: Counter, best: dict,
                   started: float, complete: bool, parity: dict | None = None) -> dict:
    du_rxids = dispatcher_du_rxids(scn)
    return {
        "status": "complete" if complete else "partial",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "C": C, "D": D, "N_RF": NRF, "PA": PA, "PA_name": "SLAPP",
            "R_min_gbps": R_MIN, "soft_qos_factor": SOFT_FACTOR,
            "physics_mode": "legacy_matlab" if scn.legacy_matlab else "corrected_phase",
            "data_dir": str(DATA), "dispatcher_selected_du_rxids": du_rxids,
            "CU_slot_to_physical_txid": [int(value) for value in ICU_LOS[:C]],
        },
        "enumeration": {
            "definition": "all labelled DU-to-CU-slot assignments with load<=N_RF; deployment=unique used slots",
            "analytical_counts_by_active_cus": {str(k): v for k, v in analytical_counts().items()},
            "analytical_total": sum(analytical_counts().values()),
            "evaluated": evaluated,
            "evaluated_by_active_cus": {str(k): counts[k] for k in range(1, C + 1)},
            "elapsed_seconds": time.perf_counter() - started,
        },
        "feasibility_counts": dict(flags),
        "cache_validation": parity,
        "best": best,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-evaluations", type=int, default=0,
                        help="0 evaluates all; positive values support smoke tests only")
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--parity-samples", type=int, default=100)
    parser.add_argument("--dry-count", action="store_true")
    args = parser.parse_args()
    expected = analytical_counts()
    print(f"Analytical feasible assignment counts: {expected}; total={sum(expected.values())}")
    if args.dry_count:
        observed = Counter(len(set(value)) for value in enumerate_capacity_feasible())
        print(f"Generator counts: {dict(observed)}; total={sum(observed.values())}")
        if dict(observed) != {k: v for k, v in expected.items() if v}:
            raise RuntimeError("Generator count does not match independent multinomial count")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = args.output_dir / "joint_enumeration_partial.json"
    final_path = args.output_dir / "joint_enumeration_final.json"
    cache_path = args.output_dir / "tier2_group_cache.json"
    scn = Scenario(str(DATA))
    du_rxids = dispatcher_du_rxids(scn)
    print(f"Pinned dispatcher physical DU rxids: {du_rxids}", flush=True)
    group_cache = build_group_cache(scn, du_rxids)
    cache_path.write_text(
        json.dumps(serializable_group_cache(group_cache, du_rxids), indent=2) + "\n",
        encoding="utf-8",
    )
    parity = validate_group_cache(scn, group_cache, args.parity_samples)
    best: dict = {}
    counts: Counter = Counter()
    flags: Counter = Counter()
    started = time.perf_counter()
    evaluated = 0
    for association in enumerate_capacity_feasible():
        if args.max_evaluations and evaluated >= args.max_evaluations:
            break
        _sinrs, rates, power = evaluate_from_group_cache(association, group_cache)
        record = candidate(association, np.asarray(rates, dtype=float), power)
        evaluated += 1
        counts[record["n_active_cus"]] += 1
        flags["soft_qos_feasible"] += int(record["soft_qos_feasible"])
        flags["exact_qos_feasible"] += int(record["exact_qos_feasible"])
        flags["strict_target_power_feasible"] += int(power["strict_target_power_feasible"])
        flags["strict_power_and_exact_qos"] += int(power["strict_target_power_feasible"] and record["exact_qos_feasible"])
        update_best(best, record)
        if evaluated % args.checkpoint_every == 0:
            payload = result_payload(scn, evaluated, counts, flags, best, started, complete=False, parity=parity)
            partial_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(f"evaluated={evaluated}/{sum(expected.values())} elapsed={payload['enumeration']['elapsed_seconds']:.1f}s")

    complete = evaluated == sum(expected.values()) and not args.max_evaluations
    payload = result_payload(scn, evaluated, counts, flags, best, started, complete=complete, parity=parity)
    destination = final_path if complete else partial_path
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {destination}")
    if complete and {k: counts[k] for k in expected} != expected:
        raise RuntimeError("Evaluated counts do not match independent analytical counts")
    if complete:
        print("Directly verifying every distinct winning assignment...", flush=True)
        verified = {}
        for name, record in best.items():
            association = tuple(record["association_cu_slots_0based"])
            if association in verified:
                continue
            direct_sinrs, direct_rates = scn._dispatch_impl(np.asarray(association, dtype=int), PA)
            direct_power = summarize_power(scn.last_diagnostics)
            cached_sinrs, cached_rates, cached_power = evaluate_from_group_cache(association, group_cache)
            if not np.allclose(direct_rates, cached_rates, rtol=1e-11, atol=1e-11):
                raise RuntimeError(f"Winning assignment rate verification failed: {association}")
            if not np.allclose(direct_sinrs, cached_sinrs, rtol=1e-11, atol=1e-11):
                raise RuntimeError(f"Winning assignment SINR verification failed: {association}")
            if direct_power != cached_power:
                raise RuntimeError(f"Winning assignment power verification failed: {association}")
            verified[association] = True
        payload["direct_winner_verification"] = {
            "status": "passed", "distinct_assignments_checked": len(verified)
        }
        final_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"winner verification PASSED for {len(verified)} distinct assignments", flush=True)


if __name__ == "__main__":
    main()
