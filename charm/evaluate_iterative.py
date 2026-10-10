"""Paired R3.1 one-shot versus iterative SLAPP on five frozen policies."""
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
OUT = HERE / "results_r3_1" / "frozen_slapp_paired"
os.environ.setdefault("CHARM_PHY_LEGACY", "1")
os.environ.setdefault("CHARM_DEPLOY_LOGIT_CLAMP", "8")
os.environ.setdefault("CHARM_RAW_NRF_LAGRANGIAN", "1")
os.environ.setdefault("CHARM_ENTROPY_WEIGHT", "1e-5")
os.environ.setdefault("CHARM_TORCH_THREADS", "1")

from charm.physical_layer import Scenario  # noqa: E402
from charm.thz_environment import THzMidhaulEnv_SE_RL  # noqa: E402
from charm.power_allocation import evaluate_iterative_slapp  # noqa: E402
from charm.evaluate_sinr import (  # noqa: E402
    METRICS, RUNS, SCENARIOS, final_checkpoint, make_agent, mean_rows,
    rate_metrics, sample_demands,
)


VARIANTS = ("slapp_one_shot", "iterative_slapp")


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

    topology_cache = {}
    group_cache = {}
    episode_rows = []
    group_iterations = []
    group_converged = []
    group_monotonic = []
    power_cv = []
    power_ratio = []
    one_shot_update_seconds = []
    iterative_update_seconds = []
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
                if association not in topology_cache:
                    topology_cache[association] = evaluate_iterative_slapp(scn, association, group_cache)
                paired = topology_cache[association]
                baseline_rates = np.asarray(paired["baseline_rates_gbps"])
                if not np.allclose(baseline_rates, np.asarray(info["rates"]), rtol=1e-11, atol=1e-11):
                    raise RuntimeError("R3.1 paired baseline differs from live PA=1 environment")
                iterative_rates = np.asarray(paired["iterative_rates_gbps"])
                row = {}
                for variant, rates in (("slapp_one_shot", baseline_rates), ("iterative_slapp", iterative_rates)):
                    for key, value in rate_metrics(rates, demands).items():
                        row[f"{variant}_{key}"] = value
                step_rows.append(row)
                for group in paired["groups"]:
                    trace = group["trace"]
                    group_iterations.append(float(group["iterations_after_one_shot"]))
                    group_converged.append(float(group["converged"]))
                    group_monotonic.append(float(group["sum_rate_monotonic"]))
                    power_cv.append(float(trace[0]["coefficient_of_variation"]))
                    power_ratio.append(float(trace[0]["max_min_positive_ratio"]))
                    one_shot_update_seconds.append(float(group["one_shot_update_seconds"]))
                    iterative_update_seconds.append(float(group["iterative_update_seconds"]))
            episode_row = {
                "train_seed": train_seed, "scenario": scenario_name,
                "eval_episode": episode, "eval_seed": eval_seed,
                "mean_demand_gbps": float(np.mean(demands)),
            }
            for variant in VARIANTS:
                for metric in METRICS:
                    episode_row[f"{variant}_{metric}"] = mean_rows(step_rows, variant, metric)
            episode_rows.append(episode_row)
        print(f"train seed {train_seed}: completed {scenario_name}", flush=True)

    with (seed_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
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
        "reviewer_item": "R3.1",
        "design": "paired fixed-policy/fixed-topology; iteration 0 is unchanged one-shot SLAPP",
        "train_seed": train_seed, "training_run": RUNS[train_seed],
        "checkpoint": str(checkpoint),
        "episodes_per_scenario": episodes, "steps_per_episode": steps,
        "scenario_summary": scenario_summary, "overall_summary": overall,
        "diagnostics": {
            "unique_projected_associations": len(topology_cache),
            "unique_physical_cu_du_groups": len(group_cache),
            "total_wall_seconds": time.perf_counter() - started,
            "mean_iterations_after_one_shot": float(np.mean(group_iterations)),
            "group_convergence_fraction": float(np.mean(group_converged)),
            "group_monotonic_fraction": float(np.mean(group_monotonic)),
            "mean_one_shot_power_cv": float(np.mean(power_cv)),
            "median_one_shot_power_max_min_ratio": float(np.median(power_ratio)),
            "mean_one_shot_update_seconds": float(np.mean(one_shot_update_seconds)),
            "mean_iterative_extra_seconds": float(np.mean(iterative_update_seconds)),
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"SAVED {summary_path}", flush=True)
    return summary_path


def aggregate(paths: list[Path]) -> Path:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    n = len(payloads)
    result = {"training_seeds": [item["train_seed"] for item in payloads], "variants": {}}
    for variant in VARIANTS:
        result["variants"][variant] = {}
        for metric in METRICS:
            values = np.array([item["overall_summary"][variant][metric] for item in payloads])
            result["variants"][variant][metric] = {
                "mean": float(np.mean(values)),
                "sample_sd": float(np.std(values, ddof=1)) if n > 1 else 0.0,
            }
    result["paired_iterative_minus_one_shot"] = {}
    for metric in METRICS:
        differences = np.array([
            item["overall_summary"]["iterative_slapp"][metric]
            - item["overall_summary"]["slapp_one_shot"][metric]
            for item in payloads
        ])
        mean = float(np.mean(differences))
        sd = float(np.std(differences, ddof=1)) if n > 1 else 0.0
        half = 2.7764451051977987 * sd / math.sqrt(n) if n == 5 else float("nan")
        result["paired_iterative_minus_one_shot"][metric] = {
            "mean": mean, "sample_sd": sd,
            "paired_95ci_low": mean - half, "paired_95ci_high": mean + half,
        }
    diagnostic_keys = [
        "mean_iterations_after_one_shot", "group_convergence_fraction",
        "group_monotonic_fraction", "mean_one_shot_power_cv",
        "median_one_shot_power_max_min_ratio", "mean_one_shot_update_seconds",
        "mean_iterative_extra_seconds",
    ]
    result["diagnostics"] = {
        key: float(np.mean([item["diagnostics"][key] for item in payloads]))
        for key in diagnostic_keys
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
