"""Paired R3.4 evaluation on the five frozen SLAPP training runs.

The environment and policy evolve under the original PA=1 trajectory. At each
step, the exact same projected association and the exact same equal-power SLNR
beamformers are scored twice: submitted SLNR-derived PA and true-SINR FP PA.
Thus the experiment isolates power allocation and does not retrain the policy.
"""
from __future__ import annotations
from charm.paths import run_root, data_directory

import argparse
import contextlib
import csv
import io
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = run_root()
HERE = ROOT / "phy"
DATA = data_directory()
OUT = HERE / "results_r3_4" / "frozen_slapp_paired"

os.environ.setdefault("CHARM_PHY_LEGACY", "1")
os.environ.setdefault("CHARM_DEPLOY_LOGIT_CLAMP", "8")
os.environ.setdefault("CHARM_RAW_NRF_LAGRANGIAN", "1")
os.environ.setdefault("CHARM_ENTROPY_WEIGHT", "1e-5")
os.environ.setdefault("CHARM_TORCH_THREADS", "1")

from charm.physical_layer import Scenario  # noqa: E402
from charm.thz_environment import THzMidhaulEnv_SE_RL  # noqa: E402
from charm.ppg_policy import JointPPG  # noqa: E402
from charm.power_allocation import evaluate_fixed_beamformers  # noqa: E402


RUNS = {
    1: "slapp_seed1",
    2: "slapp_seed2",
    3: "slapp_seed3",
    4: "slapp_seed4",
    5: "slapp_seed5",
}

SCENARIOS = {
    "Low_Demand": ((0.50, 0.25, 0.25), ((4.0, 5.5), (5.5, 6.5), (6.5, 8.0))),
    "Balanced": ((1 / 3, 1 / 3, 1 / 3), ((4.0, 5.5), (5.5, 6.5), (6.5, 8.0))),
    "High_Demand": ((0.25, 0.25, 0.50), ((4.0, 5.5), (5.5, 6.5), (6.5, 8.0))),
    "Mixed_stress": ((0.30, 0.30, 0.40), ((4.0, 5.5), (5.5, 6.5), (6.5, 8.0))),
    "Extreme_high": ((0.10, 0.20, 0.70), ((4.0, 5.5), (5.5, 6.5), (6.5, 8.0))),
}

VARIANTS = ("slapp_one_shot", "sinr_fp")
METRICS = (
    "sum_rate_gbps", "min_rate_gbps", "soft_qos_fraction",
    "exact_qos_fraction", "exact_violation_fraction", "exact_shortfall_gbps",
)


def final_checkpoint(run_name: str) -> Path:
    files = sorted((ROOT / "runs" / run_name / "checkpoints").glob("*_final.pth"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one final checkpoint for {run_name}; found {files}")
    return files[0]


def make_agent() -> JointPPG:
    return JointPPG(
        (48,), [(5,), (8, 5)], 1e-4, 1e-4, 0.99, 30, 0.15,
        noise_scale=1.0, action_std=0.25,
        action_max=0.19952623149688797, temp=2.0,
    )


def sample_demands(name: str, seed: int) -> np.ndarray:
    probabilities, ranges = SCENARIOS[name]
    rng = np.random.default_rng(seed)
    categories = rng.choice(3, size=8, p=probabilities)
    return np.array([rng.uniform(*ranges[value]) for value in categories])


def rate_metrics(rates: np.ndarray, demands: np.ndarray) -> dict:
    targets = np.minimum(6.0, demands)
    exact_failures = rates < targets
    return {
        "sum_rate_gbps": float(np.sum(rates)),
        "min_rate_gbps": float(np.min(rates)),
        "soft_qos_fraction": float(np.mean(rates >= 0.9 * targets)),
        "exact_qos_fraction": float(np.mean(~exact_failures)),
        "exact_violation_fraction": float(np.mean(exact_failures)),
        "exact_shortfall_gbps": float(np.sum(np.maximum(targets - rates, 0.0))),
    }


def mean_rows(rows: list[dict], variant: str, metric: str) -> float:
    return float(np.mean([row[f"{variant}_{metric}"] for row in rows]))


def evaluate_seed(train_seed: int, episodes: int, steps: int, force: bool) -> Path:
    seed_dir = OUT / f"train_seed_{train_seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    summary_path = seed_dir / "summary.json"
    if summary_path.exists() and not force:
        print(f"SKIP completed train seed {train_seed}", flush=True)
        return summary_path

    checkpoint = final_checkpoint(RUNS[train_seed])
    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet):
        scn = Scenario(str(DATA))
        env = THzMidhaulEnv_SE_RL(C=5, D=8, scn=scn, PA=1, NRF=3)
        agent = make_agent()
        agent.load(str(checkpoint))
        agent.policy.eval()
        agent.policy_old.eval()

    topology_cache: dict[tuple[int, ...], dict] = {}
    group_result_cache: dict = {}
    episode_rows = []
    phy_compute_seconds = 0.0
    cache_hits = 0
    cache_misses = 0
    fp_iterations = []
    fp_converged = []
    interference_to_noise = []
    started = time.perf_counter()
    for scenario_index, scenario_name in enumerate(SCENARIOS):
        for episode in range(episodes):
            eval_seed = 100_000 + scenario_index * 10_000 + episode
            demands = sample_demands(scenario_name, eval_seed)
            env.current_demands = demands.copy()
            random.seed(eval_seed)
            np.random.seed(eval_seed)
            torch.manual_seed(eval_seed)
            with contextlib.redirect_stdout(quiet):
                state = env.reset(seed=eval_seed)
            step_rows = []
            for _ in range(steps):
                with contextlib.redirect_stdout(quiet):
                    action = agent.select_action(state)
                    state, _reward, _terminated, info = env.step(action)
                agent.buffer.clear()
                association = tuple(int(value) for value in info["projected_actions"]["association_projected"])
                if association in topology_cache:
                    paired = topology_cache[association]
                    cache_hits += 1
                else:
                    phy_started = time.perf_counter()
                    paired = evaluate_fixed_beamformers(scn, association, group_result_cache)
                    phy_compute_seconds += time.perf_counter() - phy_started
                    topology_cache[association] = paired
                    cache_misses += 1
                baseline_rates = np.asarray(paired["baseline_rates_gbps"])
                if not np.allclose(baseline_rates, np.asarray(info["rates"]), rtol=1e-11, atol=1e-11):
                    raise RuntimeError("Paired baseline rates differ from the live PA=1 environment")
                fp_rates = np.asarray(paired["fp_rates_gbps"])
                row = {}
                for variant, rates in (("slapp_one_shot", baseline_rates), ("sinr_fp", fp_rates)):
                    for key, value in rate_metrics(rates, demands).items():
                        row[f"{variant}_{key}"] = value
                step_rows.append(row)
                for group in paired["groups"]:
                    fp_iterations.append(float(group["fp_iterations"]))
                    fp_converged.append(float(group["fp_converged"]))
                    interference_to_noise.append(float(group["mean_interference_to_noise"]))
            episode_row = {
                "train_seed": train_seed,
                "scenario": scenario_name,
                "eval_episode": episode,
                "eval_seed": eval_seed,
                "mean_demand_gbps": float(np.mean(demands)),
            }
            for variant in VARIANTS:
                for metric in METRICS:
                    episode_row[f"{variant}_{metric}"] = mean_rows(step_rows, variant, metric)
            episode_rows.append(episode_row)
        print(f"train seed {train_seed}: completed {scenario_name}", flush=True)

    csv_path = seed_dir / "episodes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)

    overall = {
        variant: {metric: mean_rows(episode_rows, variant, metric) for metric in METRICS}
        for variant in VARIANTS
    }
    scenario_summary = {}
    for scenario_name in SCENARIOS:
        selected = [row for row in episode_rows if row["scenario"] == scenario_name]
        scenario_summary[scenario_name] = {
            variant: {metric: mean_rows(selected, variant, metric) for metric in METRICS}
            for variant in VARIANTS
        }
    payload = {
        "reviewer_item": "R3.4",
        "design": "paired fixed-policy/fixed-topology/fixed-SLNR-beamformer comparison; baseline trajectory drives policy state",
        "train_seed": train_seed,
        "training_run": RUNS[train_seed],
        "checkpoint": str(checkpoint),
        "episodes_per_scenario": episodes,
        "steps_per_episode": steps,
        "scenario_summary": scenario_summary,
        "overall_summary": overall,
        "diagnostics": {
            "unique_projected_associations": len(topology_cache),
            "unique_physical_cu_du_groups": len(group_result_cache),
            "topology_cache_hits": cache_hits,
            "topology_cache_misses": cache_misses,
            "r3_phy_compute_seconds": phy_compute_seconds,
            "total_wall_seconds": time.perf_counter() - started,
            "mean_fp_iterations_per_cu_group": float(np.mean(fp_iterations)),
            "fp_group_convergence_fraction": float(np.mean(fp_converged)),
            "mean_interference_to_noise_ratio": float(np.mean(interference_to_noise)),
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"SAVED {summary_path}", flush=True)
    return summary_path


def aggregate(paths: list[Path]) -> Path:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    result = {"training_seeds": [item["train_seed"] for item in payloads], "variants": {}}
    n = len(payloads)
    for variant in VARIANTS:
        result["variants"][variant] = {}
        for metric in METRICS:
            values = np.array([item["overall_summary"][variant][metric] for item in payloads])
            result["variants"][variant][metric] = {
                "mean": float(np.mean(values)),
                "sample_sd": float(np.std(values, ddof=1)) if n > 1 else 0.0,
            }
    result["paired_sinr_fp_minus_slapp"] = {}
    for metric in METRICS:
        differences = np.array([
            item["overall_summary"]["sinr_fp"][metric]
            - item["overall_summary"]["slapp_one_shot"][metric]
            for item in payloads
        ])
        mean = float(np.mean(differences))
        sd = float(np.std(differences, ddof=1)) if n > 1 else 0.0
        half = 2.7764451051977987 * sd / math.sqrt(n) if n == 5 else float("nan")
        result["paired_sinr_fp_minus_slapp"][metric] = {
            "mean": mean, "sample_sd": sd,
            "paired_95ci_low": mean - half, "paired_95ci_high": mean + half,
        }
    result["diagnostics"] = {
        "mean_interference_to_noise_ratio": float(np.mean([
            item["diagnostics"]["mean_interference_to_noise_ratio"] for item in payloads
        ])),
        "mean_fp_iterations_per_cu_group": float(np.mean([
            item["diagnostics"]["mean_fp_iterations_per_cu_group"] for item in payloads
        ])),
        "fp_group_convergence_fraction": float(np.mean([
            item["diagnostics"]["fp_group_convergence_fraction"] for item in payloads
        ])),
        "total_unique_topologies_across_seed_caches": int(sum(
            item["diagnostics"]["unique_projected_associations"] for item in payloads
        )),
    }
    destination = OUT / "aggregate_five_training_seeds.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"SAVED {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--train-seed", type=int, choices=range(1, 6))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [args.train_seed] if args.train_seed else list(range(1, 6))
    paths = [evaluate_seed(seed, args.episodes, args.steps, args.force) for seed in seeds]
    if len(paths) == 5:
        aggregate(paths)


if __name__ == "__main__":
    main()
