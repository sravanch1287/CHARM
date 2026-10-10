"""Reviewer-3 physical-layer ablations without modifying the frozen PHY.

R3.4 holds the submitted equal-power SLNR beamformers fixed and replaces only
the SLNR-derived power allocation with a true-SINR sum-rate power allocator
based on the Lagrangian-dual and quadratic transforms of Shen and Yu (2018).

All helpers operate on a supplied frozen `charm_phy.Scenario`.  The original
module and the frozen training/evaluation artifacts remain byte-unchanged.
"""
from __future__ import annotations
from charm.paths import run_root, data_directory

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = run_root()

import charm.physical_layer as base  # noqa: E402


@dataclass
class FixedBeamformerGroup:
    cu_slot: int
    physical_txid: int
    members: tuple[int, ...]
    gain: np.ndarray
    h_hat: np.ndarray
    H_hat: np.ndarray
    wr_hat: np.ndarray
    w_tx: np.ndarray


def dispatcher_mapping(scn: base.Scenario, association: tuple[int, ...]) -> tuple[list[int], list[int], list[int]]:
    """Reproduce the unchanged dispatcher's physical CU/DU mapping."""
    D = len(association)
    los_count = scn.losMat.sum(axis=1)
    du_rxids = (np.argsort(los_count, kind="stable")[:D] + 1).astype(int).tolist()
    active_slots = sorted(set(int(value) for value in association))
    cu_txids = [int(base.ICU_LOS[slot]) for slot in active_slots]
    assoc_local = [active_slots.index(int(value)) for value in association]
    return du_rxids, cu_txids, assoc_local


def build_fixed_beamformer_groups(
    scn: base.Scenario, association: tuple[int, ...]
) -> tuple[list[FixedBeamformerGroup], list[int]]:
    """Build exactly the same equal-power SLNR beamformers as PA=1."""
    du_rxids, cu_txids, assoc_local = dispatcher_mapping(scn, association)
    D = len(association)
    ncu = len(cu_txids)
    npw = base.BW * base.N0_LIN

    du_bestpath: dict[int, tuple[np.ndarray, int, int]] = {}
    for j, rx in enumerate(du_rxids):
        itx = cu_txids[assoc_local[j]]
        paths = np.where((scn.rxid == rx) & (scn.txid == itx))[0]
        if len(paths) == 0:
            raise RuntimeError(f"No ray-tracing path for txid={itx}, rxid={rx}")
        strongest = int(paths[np.argmin(scn.pl[paths])])
        du_bestpath[j] = (paths, strongest, itx)

    ang_cu = np.zeros((ncu, 2))
    for local_cu in range(ncu):
        members = [j for j in range(D) if assoc_local[j] == local_cu]
        directions = []
        for j in members:
            _, strongest, _ = du_bestpath[j]
            directions.append(base.dir_vec(scn.aod_az[strongest], scn.aod_el[strongest]))
        mean_direction = np.mean(np.asarray(directions), axis=0)
        az = np.degrees(np.arctan2(mean_direction[1], mean_direction[0]))
        el = np.degrees(np.arctan2(mean_direction[2], np.hypot(mean_direction[0], mean_direction[1])))
        ang_cu[local_cu] = (el, az)

    H_all: list[np.ndarray] = [None] * D  # type: ignore[list-item]
    wrx_all: list[np.ndarray] = [None] * D  # type: ignore[list-item]
    for j in range(D):
        paths, strongest, _ = du_bestpath[j]
        local_cu = assoc_local[j]
        utx, g_cu = base.array_step(
            base._POS_CU, ang_cu[local_cu, 1], ang_cu[local_cu, 0],
            scn.aod_az[paths], scn.aod_el[paths],
        )
        urx, g_du = base.array_step(
            base._POS_DU, scn.aoa_az[strongest], scn.aoa_el[strongest],
            scn.aoa_az[paths], scn.aoa_el[paths],
        )
        gains = 10 ** (0.05 * (g_cu + g_du - scn.pl[paths]))
        H = base.calc_channel_matrix(
            utx, urx, gains, scn.delay[paths], scn.phase[paths],
            legacy_matlab=scn.legacy_matlab,
        )
        strongest_local = int(np.argmin(scn.pl[paths]))
        wrx = np.conj(urx[:, strongest_local])
        wrx /= np.linalg.norm(wrx)
        H_all[j] = H
        wrx_all[j] = wrx

    groups = []
    for local_cu, physical_txid in enumerate(cu_txids):
        members = tuple(j for j in range(D) if assoc_local[j] == local_cu)
        K = len(members)
        H_hat = np.stack([H_all[j] for j in members], axis=2)
        wr_hat = np.column_stack([wrx_all[j] for j in members])
        h_hat = np.array([
            wr_hat[:, k].conj() @ H_hat[:, :, k] for k in range(K)
        ])
        equal_power = np.full(K, base.P_TX_LIN / K)
        w_tx = base.leakage_precoding(h_hat, npw, equal_power)
        gain = np.empty((K, K), dtype=float)
        for receiver in range(K):
            for stream in range(K):
                gain[receiver, stream] = float(np.abs(
                    wr_hat[:, receiver] @ H_hat[:, :, receiver] @ w_tx[:, stream]
                ) ** 2)
        groups.append(FixedBeamformerGroup(
            cu_slot=sorted(set(association))[local_cu],
            physical_txid=physical_txid,
            members=members,
            gain=gain,
            h_hat=h_hat,
            H_hat=H_hat,
            wr_hat=wr_hat,
            w_tx=w_tx,
        ))
    return groups, du_rxids


def sinr_from_gain(gain: np.ndarray, power: np.ndarray, noise: float) -> np.ndarray:
    signal = np.diag(gain) * power
    total = gain @ power + noise
    interference_noise = np.maximum(total - signal, np.finfo(float).tiny)
    return signal / interference_noise


def uncapped_sum_se(gain: np.ndarray, power: np.ndarray, noise: float) -> float:
    return float(np.sum(np.log2(1.0 + sinr_from_gain(gain, power, noise))))


def capped_rates(sinr: np.ndarray) -> np.ndarray:
    return base.BW * (1.0 - base.BW_LOSS) * np.minimum(
        np.log2(1.0 + sinr), base.MAX_SE
    ) / 1e9


def gain_for_beamformers(H_hat: np.ndarray, wr_hat: np.ndarray, w_tx: np.ndarray) -> np.ndarray:
    K = H_hat.shape[2]
    gain = np.empty((K, K), dtype=float)
    for receiver in range(K):
        for stream in range(K):
            gain[receiver, stream] = float(np.abs(
                wr_hat[:, receiver] @ H_hat[:, :, receiver] @ w_tx[:, stream]
            ) ** 2)
    return gain


def solve_sinr_fp(
    gain: np.ndarray,
    noise: float,
    power_budget: float = base.P_TX_LIN,
    max_iter: int = 1000,
    tolerance: float = 1e-8,
    initial_power: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Direct FP for unweighted true-SINR sum-SE with a sum-power constraint.

    The nonconvex problem is solved to a stationary point. Multiple deterministic
    initializations are handled by the caller when desired.
    """
    K = gain.shape[0]
    if gain.shape != (K, K):
        raise ValueError("gain must be square")
    if K == 1:
        power = np.array([power_budget], dtype=float)
        return power, {"iterations": 0, "converged": True, "objective_history": [uncapped_sum_se(gain, power, noise)]}
    if initial_power is None:
        power = np.full(K, power_budget / K, dtype=float)
    else:
        power = np.maximum(np.asarray(initial_power, dtype=float), 0.0)
        if power.sum() <= 0:
            power[:] = power_budget / K
        elif power.sum() > power_budget:
            power *= power_budget / power.sum()

    history = [uncapped_sum_se(gain, power, noise)]
    converged = False
    for iteration in range(1, max_iter + 1):
        x = np.sqrt(np.maximum(power, 0.0))
        sinr = sinr_from_gain(gain, power, noise)
        gamma = sinr
        total_received = gain @ power + noise
        y = np.sqrt((1.0 + gamma) * np.diag(gain)) * x / np.maximum(
            total_received, np.finfo(float).tiny
        )
        quadratic = gain.T @ (y ** 2)
        linear = y * np.sqrt((1.0 + gamma) * np.diag(gain))

        def candidate(lagrange: float) -> np.ndarray:
            return linear / np.maximum(quadratic + lagrange, np.finfo(float).tiny)

        x_new = candidate(0.0)
        if float(x_new @ x_new) > power_budget:
            low, high = 0.0, 1.0
            while float(candidate(high) @ candidate(high)) > power_budget:
                high *= 2.0
            for _ in range(100):
                middle = 0.5 * (low + high)
                trial = candidate(middle)
                if float(trial @ trial) > power_budget:
                    low = middle
                else:
                    high = middle
            x_new = candidate(high)
        power_new = x_new ** 2
        objective = uncapped_sum_se(gain, power_new, noise)
        history.append(objective)
        if objective + 1e-9 < history[-2]:
            raise RuntimeError(
                f"FP objective decreased from {history[-2]} to {objective}"
            )
        power = power_new
        if abs(history[-1] - history[-2]) <= tolerance * max(1.0, abs(history[-2])):
            converged = True
            break
    return power, {
        "iterations": iteration,
        "converged": converged,
        "objective_history": history,
        "sum_power_w": float(power.sum()),
    }


def brute_force_power_grid(
    gain: np.ndarray,
    noise: float,
    resolution: int = 1000,
) -> tuple[np.ndarray, float]:
    """Independent dense-simplex reference for K<=3 (uncapped sum-SE)."""
    K = gain.shape[0]
    budget = base.P_TX_LIN
    if K == 1:
        power = np.array([budget])
        return power, uncapped_sum_se(gain, power, noise)
    best_power = None
    best_objective = -math.inf
    if K == 2:
        first = np.linspace(0.0, budget, resolution + 1)
        candidates = np.column_stack((first, budget - first))
        signal = candidates * np.diag(gain)[None, :]
        total = candidates @ gain.T + noise
        sinr = signal / np.maximum(total - signal, np.finfo(float).tiny)
        objective = np.sum(np.log2(1.0 + sinr), axis=1)
        index = int(np.argmax(objective))
        return candidates[index], float(objective[index])
    if K != 3:
        raise ValueError("Dense reference is intentionally limited to K<=3")
    quantum = budget / resolution
    for first_index in range(resolution + 1):
        second_index = np.arange(resolution - first_index + 1)
        candidates = np.column_stack((
            np.full(len(second_index), first_index * quantum),
            second_index * quantum,
            (resolution - first_index - second_index) * quantum,
        ))
        signal = candidates * np.diag(gain)[None, :]
        total = candidates @ gain.T + noise
        sinr = signal / np.maximum(total - signal, np.finfo(float).tiny)
        objective = np.sum(np.log2(1.0 + sinr), axis=1)
        index = int(np.argmax(objective))
        if float(objective[index]) > best_objective:
            best_objective = float(objective[index])
            best_power = candidates[index].copy()
    assert best_power is not None
    return best_power, best_objective


def solve_sinr_fp_multistart(gain: np.ndarray, noise: float) -> tuple[np.ndarray, dict]:
    """Use deterministic equal, corner-biased, and SLAPP-like starts."""
    K = gain.shape[0]
    starts = [np.full(K, base.P_TX_LIN / K)]
    for dominant in range(K):
        start = np.full(K, 0.05 * base.P_TX_LIN / max(K - 1, 1))
        start[dominant] = 0.95 * base.P_TX_LIN
        starts.append(start)
    solutions = []
    for index, start in enumerate(starts):
        power, diagnostics = solve_sinr_fp(gain, noise, initial_power=start)
        solutions.append((uncapped_sum_se(gain, power, noise), index, power, diagnostics))
    objective, start_index, power, diagnostics = max(solutions, key=lambda item: item[0])
    diagnostics = dict(diagnostics)
    diagnostics.update({
        "selected_start": start_index,
        "starts_tested": len(starts),
        "uncapped_sum_se": objective,
        "all_start_objectives": [item[0] for item in solutions],
    })
    return power, diagnostics


def power_nonuniformity(power: np.ndarray) -> dict:
    positive = power[power > np.finfo(float).tiny]
    return {
        "coefficient_of_variation": float(np.std(power) / max(np.mean(power), np.finfo(float).tiny)),
        "max_min_positive_ratio": float(np.max(positive) / np.min(positive)) if len(positive) else math.inf,
    }


def solve_iterative_slapp_group(
    group: FixedBeamformerGroup,
    legacy_abs: bool,
    max_iter: int = 10,
    tolerance_gbps: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Alternate leakage beamforming and submitted SLAPP PA for one CU.

    Trace index 0 is the current submitted one-shot result: W is computed under
    equal power, followed by one SLAPP power update. Subsequent entries recompute
    W using the preceding nonuniform power vector before updating P again.
    """
    noise = base.BW * base.N0_LIN
    one_shot_started = time.perf_counter()
    power, _ = base.poweralloc_slapp(
        group.H_hat, group.wr_hat, group.w_tx, noise,
        legacy_abs=legacy_abs, return_diagnostics=True,
    )
    w_tx = group.w_tx.copy()
    gain = gain_for_beamformers(group.H_hat, group.wr_hat, w_tx)
    sinr = sinr_from_gain(gain, power, noise)
    rates = capped_rates(sinr)
    one_shot_seconds = time.perf_counter() - one_shot_started
    nonuniformity = power_nonuniformity(power)
    trace = [{
        "iteration": 0,
        "sum_rate_gbps": float(np.sum(rates)),
        "min_rate_gbps": float(np.min(rates)),
        "power_w": power.tolist(),
        **nonuniformity,
    }]
    converged = False
    monotonic = True
    iterative_started = time.perf_counter()
    for iteration in range(1, max_iter + 1):
        w_next = base.leakage_precoding(group.h_hat, noise, power)
        power_next, _ = base.poweralloc_slapp(
            group.H_hat, group.wr_hat, w_next, noise,
            legacy_abs=legacy_abs, return_diagnostics=True,
        )
        gain_next = gain_for_beamformers(group.H_hat, group.wr_hat, w_next)
        sinr_next = sinr_from_gain(gain_next, power_next, noise)
        rates_next = capped_rates(sinr_next)
        sum_rate = float(np.sum(rates_next))
        monotonic = monotonic and sum_rate + 1e-12 >= trace[-1]["sum_rate_gbps"]
        trace.append({
            "iteration": iteration,
            "sum_rate_gbps": sum_rate,
            "min_rate_gbps": float(np.min(rates_next)),
            "power_w": power_next.tolist(),
            **power_nonuniformity(power_next),
        })
        delta = abs(sum_rate - trace[-2]["sum_rate_gbps"])
        w_tx, power, gain, sinr, rates = w_next, power_next, gain_next, sinr_next, rates_next
        if delta < tolerance_gbps:
            converged = True
            break
    iterative_seconds = time.perf_counter() - iterative_started
    return sinr, rates, {
        "trace": trace,
        "iterations_after_one_shot": len(trace) - 1,
        "converged": converged,
        "sum_rate_monotonic": monotonic,
        "one_shot_update_seconds": one_shot_seconds,
        "iterative_update_seconds": iterative_seconds,
        "tolerance_gbps": tolerance_gbps,
        "max_iter": max_iter,
    }


def evaluate_iterative_slapp(
    scn: base.Scenario,
    association: tuple[int, ...],
    group_result_cache: dict | None = None,
) -> dict:
    """Paired submitted one-shot versus iterative SLAPP for one association."""
    desired_keys = [
        (cu_slot, tuple(index for index, value in enumerate(association) if value == cu_slot))
        for cu_slot in sorted(set(association))
    ]
    if group_result_cache is None:
        group_result_cache = {}
    if any(key not in group_result_cache for key in desired_keys):
        groups, du_rxids = build_fixed_beamformer_groups(scn, association)
        for group in groups:
            key = (group.cu_slot, group.members)
            if key in group_result_cache:
                continue
            noise = base.BW * base.N0_LIN
            baseline_power, _ = base.poweralloc_slapp(
                group.H_hat, group.wr_hat, group.w_tx, noise,
                legacy_abs=scn.legacy_matlab, return_diagnostics=True,
            )
            baseline_sinrs = sinr_from_gain(group.gain, baseline_power, noise)
            iterative_sinrs, iterative_rates, diagnostics = solve_iterative_slapp_group(
                group, legacy_abs=scn.legacy_matlab
            )
            group_result_cache[key] = {
                "baseline_sinrs": baseline_sinrs,
                "baseline_rates": capped_rates(baseline_sinrs),
                "iterative_sinrs": iterative_sinrs,
                "iterative_rates": iterative_rates,
                "diagnostics": {
                    "cu_slot": group.cu_slot,
                    "physical_txid": group.physical_txid,
                    "members": list(group.members),
                    **diagnostics,
                },
            }
    else:
        du_rxids, _, _ = dispatcher_mapping(scn, association)

    D = len(association)
    baseline_rates = np.zeros(D)
    baseline_sinrs = np.zeros(D)
    iterative_rates = np.zeros(D)
    iterative_sinrs = np.zeros(D)
    records = []
    for key in desired_keys:
        _, members = key
        entry = group_result_cache[key]
        baseline_rates[list(members)] = entry["baseline_rates"]
        baseline_sinrs[list(members)] = entry["baseline_sinrs"]
        iterative_rates[list(members)] = entry["iterative_rates"]
        iterative_sinrs[list(members)] = entry["iterative_sinrs"]
        records.append(entry["diagnostics"])
    direct_sinrs, direct_rates = scn._dispatch_impl(np.asarray(association, dtype=int), 1)
    if not np.allclose(direct_rates, baseline_rates, rtol=1e-11, atol=1e-11):
        raise RuntimeError("R3.1 one-shot reconstruction differs from unchanged dispatcher rates")
    if not np.allclose(direct_sinrs, baseline_sinrs, rtol=1e-9, atol=1e-11):
        raise RuntimeError("R3.1 one-shot reconstruction differs from unchanged dispatcher SINRs")
    return {
        "association": list(association),
        "du_rxids": du_rxids,
        "baseline_rates_gbps": baseline_rates,
        "baseline_sinrs": baseline_sinrs,
        "iterative_rates_gbps": iterative_rates,
        "iterative_sinrs": iterative_sinrs,
        "groups": records,
    }


def evaluate_fixed_beamformers(
    scn: base.Scenario,
    association: tuple[int, ...],
    group_result_cache: dict | None = None,
) -> dict:
    """Paired PA=1 versus SINR-FP evaluation for one full association."""
    desired_keys = [
        (cu_slot, tuple(index for index, value in enumerate(association) if value == cu_slot))
        for cu_slot in sorted(set(association))
    ]
    if group_result_cache is None:
        group_result_cache = {}
    if any(key not in group_result_cache for key in desired_keys):
        built_groups, du_rxids = build_fixed_beamformer_groups(scn, association)
    else:
        built_groups = []
        du_rxids, _, _ = dispatcher_mapping(scn, association)
    D = len(association)
    noise = base.BW * base.N0_LIN
    for group in built_groups:
        key = (group.cu_slot, group.members)
        if key in group_result_cache:
            continue
        K = len(group.members)
        baseline_power, baseline_power_diag = base.poweralloc_slapp(
            group.H_hat, group.wr_hat, group.w_tx, noise,
            legacy_abs=scn.legacy_matlab, return_diagnostics=True,
        )
        baseline_group_sinr = sinr_from_gain(group.gain, baseline_power, noise)
        fp_power, fp_diag = solve_sinr_fp_multistart(group.gain, noise)
        fp_group_sinr = sinr_from_gain(group.gain, fp_power, noise)
        interference = group.gain @ fp_power - np.diag(group.gain) * fp_power
        record = {
            "cu_slot": group.cu_slot,
            "physical_txid": group.physical_txid,
            "members": list(group.members),
            "baseline_power_w": baseline_power.tolist(),
            "fp_power_w": fp_power.tolist(),
            "baseline_sum_power_w": float(baseline_power.sum()),
            "fp_sum_power_w": float(fp_power.sum()),
            "baseline_uncapped_sum_se": uncapped_sum_se(group.gain, baseline_power, noise),
            "fp_uncapped_sum_se": uncapped_sum_se(group.gain, fp_power, noise),
            "fp_iterations": fp_diag["iterations"],
            "fp_converged": fp_diag["converged"],
            "fp_selected_start": fp_diag["selected_start"],
            "mean_interference_to_noise": float(np.mean(interference / noise)),
            "baseline_power_diagnostics": {
                "finite_target_feasible": bool(np.all(baseline_power_diag["finite_target_feasible"])),
                "budget_infeasible": bool(baseline_power_diag["budget_infeasible"]),
                "under_target_after_normalization": bool(np.any(baseline_power_diag["under_target_after_normalization"])),
            },
        }
        if baseline_power.sum() > base.P_TX_LIN * (1.0 + 1e-10):
            raise RuntimeError("Baseline SLAPP exceeded the CU power budget")
        if fp_power.sum() > base.P_TX_LIN * (1.0 + 1e-10):
            raise RuntimeError("SINR-FP exceeded the CU power budget")
        group_result_cache[key] = {
            "baseline_sinrs": baseline_group_sinr.copy(),
            "baseline_rates": capped_rates(baseline_group_sinr),
            "fp_sinrs": fp_group_sinr.copy(),
            "fp_rates": capped_rates(fp_group_sinr),
            "record": record,
        }

    baseline_rates = np.zeros(D)
    baseline_sinrs = np.zeros(D)
    fp_rates = np.zeros(D)
    fp_sinrs = np.zeros(D)
    group_records = []
    for key in desired_keys:
        _, members = key
        entry = group_result_cache[key]
        baseline_sinrs[list(members)] = entry["baseline_sinrs"]
        baseline_rates[list(members)] = entry["baseline_rates"]
        fp_sinrs[list(members)] = entry["fp_sinrs"]
        fp_rates[list(members)] = entry["fp_rates"]
        group_records.append(entry["record"])

    direct_sinrs, direct_rates = scn._dispatch_impl(np.asarray(association, dtype=int), 1)
    if not np.allclose(direct_rates, baseline_rates, rtol=1e-11, atol=1e-11):
        raise RuntimeError(
            f"Rebuilt PA=1 rates differ from unchanged dispatcher; max error="
            f"{np.max(np.abs(np.asarray(direct_rates) - baseline_rates))}"
        )
    if not np.allclose(direct_sinrs, baseline_sinrs, rtol=1e-9, atol=1e-11):
        raise RuntimeError(
            f"Rebuilt PA=1 SINRs differ from unchanged dispatcher for {association}; "
            f"max error={np.max(np.abs(np.asarray(direct_sinrs) - baseline_sinrs))}; "
            f"direct={np.asarray(direct_sinrs).tolist()}; rebuilt={baseline_sinrs.tolist()}"
        )
    return {
        "association": list(association),
        "du_rxids": du_rxids,
        "baseline_rates_gbps": baseline_rates,
        "baseline_sinrs": baseline_sinrs,
        "fp_rates_gbps": fp_rates,
        "fp_sinrs": fp_sinrs,
        "groups": group_records,
    }
