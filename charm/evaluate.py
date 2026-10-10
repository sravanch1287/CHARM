"""Sequential, restartable evaluation of the 15 frozen CHARM models.

Protocol corrections relative to the old SERL test scripts:
* all five paper demand mixtures;
* fresh probabilistically sampled demand vector each episode;
* demand installed before env.reset();
* 80 steps/episode;
* true joint raw feasibility reported separately from the legacy product of
  marginal success rates;
* common evaluation randomness across independently trained models;
* one JSON and one episode CSV saved immediately after each model.
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
from pathlib import Path

import numpy as np
import torch


ROOT = run_root()
DATA = data_directory()
OUT = ROOT / "evaluation" / "five_demand_five_seed"

os.environ.setdefault("CHARM_PHY_LEGACY", "1")
os.environ.setdefault("CHARM_DEPLOY_LOGIT_CLAMP", "8")
os.environ.setdefault("CHARM_RAW_NRF_LAGRANGIAN", "1")
os.environ.setdefault("CHARM_ENTROPY_WEIGHT", "1e-5")
os.environ.setdefault("CHARM_TORCH_THREADS", "1")

from charm.physical_layer import Scenario  # noqa: E402
from charm.thz_environment import THzMidhaulEnv_SE_RL  # noqa: E402
from charm.ppg_policy import JointPPG  # noqa: E402


METHODS = {
    "SLAPP": {
        "pa": 1,
        "color": "#FF0000",
        "runs": {
            1: "slapp_seed1",
            2: "slapp_seed2",
            3: "slapp_seed3",
            # User selected the independently rerun seed-4 trial 2.
            4: "slapp_seed4",
            5: "slapp_seed5",
        },
    },
    "EPA": {
        "pa": 4,
        "color": "#0000FF",
        "runs": {1: "epa_seed1", **{s: f"epa_seed{s}" for s in range(2, 6)}},
    },
    "DPA2": {
        "pa": 3,
        "color": "#008000",
        "runs": {1: "dpa2_seed1", **{s: f"dpa2_seed{s}" for s in range(2, 6)}},
    },
}

SCENARIOS = {
    "Low_Demand": ((0.50, 0.25, 0.25), ((4.0, 5.5), (5.5, 6.5), (6.5, 8.0))),
    "Balanced": ((1 / 3, 1 / 3, 1 / 3), ((4.0, 5.5), (5.5, 6.5), (6.5, 8.0))),
    "High_Demand": ((0.25, 0.25, 0.50), ((4.0, 5.5), (5.5, 6.5), (6.5, 8.0))),
    "Mixed_stress": ((0.30, 0.30, 0.40), ((4.0, 5.5), (5.5, 6.5), (6.5, 8.0))),
    "Extreme_high": ((0.10, 0.20, 0.70), ((4.0, 5.5), (5.5, 6.5), (6.5, 8.0))),
}


def final_checkpoint(run_name: str) -> Path:
    files = sorted((ROOT / "runs" / run_name / "checkpoints").glob("*_final.pth"))
    if len(files) != 1:
        raise RuntimeError(f"Expected exactly one final checkpoint for {run_name}, found {files}")
    return files[0]


def make_agent() -> JointPPG:
    return JointPPG(
        (48,), [(5,), (8, 5)], 1e-4, 1e-4, 0.99, 30, 0.15,
        noise_scale=1.0, action_std=0.25, action_max=0.19952623149688797, temp=2.0,
    )


def sample_demands(scenario: str, eval_seed: int) -> tuple[np.ndarray, np.ndarray]:
    probs, ranges = SCENARIOS[scenario]
    rng = np.random.default_rng(eval_seed)
    cats = rng.choice(3, size=8, p=probs)
    demands = np.array([rng.uniform(*ranges[c]) for c in cats], dtype=np.float64)
    return demands, cats


def mean_dict(rows: list[dict], keys: list[str]) -> dict:
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


METRICS = [
    "sum_rate_gbps", "qos_satisfaction", "spectral_efficiency",
    "safety_success", "validity_success", "nrf_success", "joint_raw_success",
    "legacy_product_success", "nrf_fixes", "dus_moved", "safety_added",
    "validity_reassignments", "deployment_changed", "association_changed",
    "rate_violation", "raw_cus", "projected_cus", "reward",
]


def evaluate_model(method: str, train_seed: int, episodes: int, steps: int, force: bool) -> Path:
    spec = METHODS[method]
    model_id = f"{method.lower()}_trainseed{train_seed}"
    model_dir = OUT / "models" / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    result_path = model_dir / "summary.json"
    if result_path.exists() and not force:
        print(f"SKIP completed {model_id}", flush=True)
        return result_path

    checkpoint = final_checkpoint(spec["runs"][train_seed])
    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet):
        scenario_phy = Scenario(str(DATA))
        env = THzMidhaulEnv_SE_RL(C=5, D=8, scn=scenario_phy, PA=spec["pa"], NRF=3)
        agent = make_agent()
        agent.load(str(checkpoint))
        agent.policy.eval()
        agent.policy_old.eval()

    episode_rows: list[dict] = []
    category_counts = np.zeros(3, dtype=np.int64)
    for scenario_index, scenario_name in enumerate(SCENARIOS):
        for episode in range(episodes):
            # Identical demand and policy-sampling seeds for every trained model.
            eval_seed = 100_000 + scenario_index * 10_000 + episode
            demands, cats = sample_demands(scenario_name, eval_seed)
            category_counts += np.bincount(cats, minlength=3)
            env.current_demands = demands.copy()
            random.seed(eval_seed)
            np.random.seed(eval_seed)
            torch.manual_seed(eval_seed)
            with contextlib.redirect_stdout(quiet):
                state = env.reset(seed=eval_seed)
            step_rows: list[dict] = []
            for _ in range(steps):
                with contextlib.redirect_stdout(quiet):
                    action = agent.select_action(state)
                    state, reward, _terminated, info = env.step(action)
                # Evaluation must not accumulate a training rollout.
                agent.buffer.clear()
                projection = info["projection_info"]
                safety = float(projection["safety_deployments_added"] == 0)
                validity = float(projection["validity_reassignments"] == 0)
                nrf = float(projection["nrf_violations_fixed"] == 0)
                raw_cus = float(np.sum(info["raw_actions"]["x_i_raw"]))
                projected_cus = float(np.sum(info["projected_actions"]["x_i_projected"]))
                step_rows.append({
                    "sum_rate_gbps": float(info["reward_components"]["sum_rates"]),
                    "qos_satisfaction": float(info["qos_satisfaction"]),
                    "spectral_efficiency": float(info["spectral_efficiency"]),
                    "safety_success": safety,
                    "validity_success": validity,
                    "nrf_success": nrf,
                    "joint_raw_success": safety * validity * nrf,
                    "legacy_product_success": 0.0,  # filled from episode marginals below
                    "nrf_fixes": float(projection["nrf_violations_fixed"]),
                    "dus_moved": float(projection["dus_moved_for_nrf"]),
                    "safety_added": float(projection["safety_deployments_added"]),
                    "validity_reassignments": float(projection["validity_reassignments"]),
                    "deployment_changed": float(bool(projection["deployment_changed"])),
                    "association_changed": float(bool(projection["association_changed"])),
                    "rate_violation": float(info["rate_viol"]),
                    "raw_cus": raw_cus,
                    "projected_cus": projected_cus,
                    "reward": float(reward),
                })
            episode_row = mean_dict(step_rows, METRICS)
            episode_row["legacy_product_success"] = (
                episode_row["safety_success"] * episode_row["validity_success"] * episode_row["nrf_success"]
            )
            episode_row.update({
                "method": method, "train_seed": train_seed, "scenario": scenario_name,
                "eval_episode": episode, "eval_seed": eval_seed,
                "mean_demand": float(np.mean(demands)),
                "low_count": int(np.sum(cats == 0)), "medium_count": int(np.sum(cats == 1)),
                "high_count": int(np.sum(cats == 2)),
            })
            episode_rows.append(episode_row)
        print(f"{model_id}: completed {scenario_name}", flush=True)

    csv_path = model_dir / "episodes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)

    scenario_summary = {}
    for scenario_name in SCENARIOS:
        selected = [row for row in episode_rows if row["scenario"] == scenario_name]
        scenario_summary[scenario_name] = mean_dict(selected, METRICS)
    payload = {
        "method": method, "pa": spec["pa"], "color": spec["color"], "train_seed": train_seed,
        "training_run": spec["runs"][train_seed], "checkpoint": str(checkpoint),
        "evaluation_episodes_per_scenario": episodes, "steps_per_episode": steps,
        "qos_rule": "rate >= 0.9*min(6,demand)",
        "demand_sampling": "fresh categorical sample per DU per episode",
        "category_counts_all_scenarios": category_counts.tolist(),
        "scenario_summary": scenario_summary,
        "overall_summary": mean_dict(episode_rows, METRICS),
    }
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"SAVED {result_path}", flush=True)
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--method", choices=["SLAPP", "EPA", "DPA2"])
    parser.add_argument("--train-seed", type=int, choices=range(1, 6))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    methods = [args.method] if args.method else ["SLAPP", "EPA", "DPA2"]
    seeds = [args.train_seed] if args.train_seed else list(range(1, 6))
    for method in methods:
        for train_seed in seeds:
            evaluate_model(method, train_seed, args.episodes, args.steps, args.force)


if __name__ == "__main__":
    main()
