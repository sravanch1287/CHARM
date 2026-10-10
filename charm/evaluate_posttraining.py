"""Regenerate Table 5 under one common post-training protocol.

Five independently trained models per original method are evaluated for 50
episodes under the training demand distribution U[4,8]. For the five frozen
SLAPP models, identical projected topologies are additionally scored with
SINR-FP and iterative SLAPP. No optimizer step or retraining occurs.
"""
from __future__ import annotations
from charm.paths import run_root, data_directory

import contextlib
import csv
import io
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = run_root()
HERE = ROOT / "phy"
DATA = data_directory()
OUT = Path(os.environ.get("CHARM_TABLE5_OUT", str(HERE / "table5_posttraining_u4_8")))
os.environ.setdefault("CHARM_PHY_LEGACY", "1")
os.environ.setdefault("CHARM_DEPLOY_LOGIT_CLAMP", "8")
os.environ.setdefault("CHARM_RAW_NRF_LAGRANGIAN", "1")
os.environ.setdefault("CHARM_ENTROPY_WEIGHT", "1e-5")
os.environ.setdefault("CHARM_TORCH_THREADS", "1")

from charm.physical_layer import Scenario  # noqa: E402
from charm.thz_environment import THzMidhaulEnv_SE_RL  # noqa: E402
from charm.power_allocation import evaluate_fixed_beamformers, evaluate_iterative_slapp  # noqa: E402


RUNS = {
    "SLAPP": {
        "pa": 1,
        "runs": {
            1: "slapp_seed1",
            2: "slapp_seed2", 3: "slapp_seed3",
            4: "slapp_seed4", 5: "slapp_seed5",
        },
    },
    "EPA": {
        "pa": 4,
        "runs": {1: "epa_seed1", **{s: f"epa_seed{s}" for s in range(2, 6)}},
    },
    "DPA2": {
        "pa": 3,
        "runs": {1: "dpa2_seed1", **{s: f"dpa2_seed{s}" for s in range(2, 6)}},
    },
}

VARIANT_ORDER = ("EPA", "DPA2", "SLAPP", "SINR_FP", "SLAPP_ITER")
METRICS = (
    "convergence_score", "sum_rate_gbps", "qos_satisfaction",
    "joint_raw_success", "nrf_activity", "dus_moved",
)


def new_agent():
    from legacy_policy import JointPPG
    return JointPPG(
        (48,), [(5,), (8, 5)], 1e-4, 1e-4, 0.99, 30, 0.15,
        noise_scale=1.0, action_std=0.25,
        action_max=0.19952623149688797, temp=2.0,
    )


def checkpoint_for(method: str, seed: int) -> Path:
    run_name = RUNS[method]["runs"][seed]
    files = sorted((ROOT / "runs" / run_name / "checkpoints").glob("*_final.pth"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one checkpoint for {run_name}; found {files}")
    return files[0]


def episode_summary(step_rows: list[dict], qos_key: str) -> dict:
    safety_added = float(np.mean([row["safety_added"] for row in step_rows]))
    validity_reassign = float(np.mean([row["validity_reassign"] for row in step_rows]))
    nrf_activity = float(np.mean([row["nrf_activity"] for row in step_rows]))
    qos = float(np.mean([row[qos_key] for row in step_rows]))
    convergence = (
        (1.0 - min(1.0, safety_added / 3.0))
        * (1.0 - min(1.0, validity_reassign / 8.0))
        * (1.0 - min(1.0, nrf_activity / 5.0))
        * qos
    )
    return {
        "convergence_score": convergence,
        "qos_satisfaction": qos,
        "nrf_activity": nrf_activity,
        "dus_moved": float(np.mean([row["dus_moved"] for row in step_rows])),
        "joint_raw_success": float(np.mean([row["joint_raw_success"] for row in step_rows])),
    }


def evaluate_model(method: str, train_seed: int, episodes: int = 50, steps: int = 80) -> dict:
    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet):
        scn = Scenario(str(DATA))
        env = THzMidhaulEnv_SE_RL(C=5, D=8, scn=scn, PA=RUNS[method]["pa"], NRF=3)
        agent = new_agent()
        checkpoint = checkpoint_for(method, train_seed)
        agent.load(str(checkpoint))
        agent.policy.eval(); agent.policy_old.eval()
    fp_topology_cache = {}; fp_group_cache = {}
    iter_topology_cache = {}; iter_group_cache = {}
    episode_rows = []
    iterative_step_deltas = []
    iterative_link_abs_deltas = []
    iterative_qos_changed_steps = 0
    for episode in range(episodes):
        eval_seed = 500_000 + episode
        demands = np.random.default_rng(eval_seed).uniform(4.0, 8.0, size=8)
        env.current_demands = demands.copy()
        random.seed(eval_seed); np.random.seed(eval_seed); torch.manual_seed(eval_seed)
        with contextlib.redirect_stdout(quiet):
            state = env.reset(seed=eval_seed)
        base_steps = []; fp_steps = []; iter_steps = []
        for _ in range(steps):
            with contextlib.redirect_stdout(quiet):
                action = agent.select_action(state)
                state, _reward, _terminated, info = env.step(action)
            agent.buffer.clear()
            projection = info["projection_info"]
            safety = float(projection["safety_deployments_added"] == 0)
            validity = float(projection["validity_reassignments"] == 0)
            nrf = float(projection["nrf_violations_fixed"] == 0)
            common = {
                "safety_added": float(projection["safety_deployments_added"]),
                "validity_reassign": float(projection["validity_reassignments"]),
                "nrf_activity": float(projection["nrf_violations_fixed"]),
                "dus_moved": float(projection["dus_moved_for_nrf"]),
                "joint_raw_success": safety * validity * nrf,
            }
            targets = np.minimum(6.0, demands)
            rates = np.asarray(info["rates"])
            base_steps.append({
                **common,
                "qos": float(np.mean(rates >= 0.9 * targets)),
                "sum_rate_gbps": float(np.sum(rates)),
            })
            if method == "SLAPP":
                association = tuple(int(value) for value in info["projected_actions"]["association_projected"])
                if association not in fp_topology_cache:
                    fp_topology_cache[association] = evaluate_fixed_beamformers(scn, association, fp_group_cache)
                if association not in iter_topology_cache:
                    iter_topology_cache[association] = evaluate_iterative_slapp(scn, association, iter_group_cache)
                fp_rates = np.asarray(fp_topology_cache[association]["fp_rates_gbps"])
                iter_rates = np.asarray(iter_topology_cache[association]["iterative_rates_gbps"])
                iterative_step_deltas.append(float(np.sum(iter_rates) - np.sum(rates)))
                iterative_link_abs_deltas.extend(np.abs(iter_rates - rates).tolist())
                iterative_qos_changed_steps += int(
                    not np.array_equal(iter_rates >= 0.9 * targets, rates >= 0.9 * targets)
                )
                fp_steps.append({**common, "qos": float(np.mean(fp_rates >= 0.9 * targets)), "sum_rate_gbps": float(np.sum(fp_rates))})
                iter_steps.append({**common, "qos": float(np.mean(iter_rates >= 0.9 * targets)), "sum_rate_gbps": float(np.sum(iter_rates))})

        base_summary = episode_summary(base_steps, "qos")
        base_summary["sum_rate_gbps"] = float(np.mean([row["sum_rate_gbps"] for row in base_steps]))
        row = {"episode": episode, method: base_summary}
        if method == "SLAPP":
            fp_summary = episode_summary(fp_steps, "qos")
            fp_summary["sum_rate_gbps"] = float(np.mean([item["sum_rate_gbps"] for item in fp_steps]))
            iter_summary = episode_summary(iter_steps, "qos")
            iter_summary["sum_rate_gbps"] = float(np.mean([item["sum_rate_gbps"] for item in iter_steps]))
            row["SINR_FP"] = fp_summary; row["SLAPP_ITER"] = iter_summary
        episode_rows.append(row)
    variants = [method] + (["SINR_FP", "SLAPP_ITER"] if method == "SLAPP" else [])
    summaries = {
        variant: {
            metric: float(np.mean([row[variant][metric] for row in episode_rows]))
            for metric in METRICS
        }
        for variant in variants
    }
    diagnostics = None
    if method == "SLAPP":
        group_records = [entry["diagnostics"] for entry in iter_group_cache.values()]
        changed_power_groups = 0
        maximum_power_change_w = 0.0
        for record in group_records:
            first = np.asarray(record["trace"][0]["power_w"], dtype=float)
            final = np.asarray(record["trace"][-1]["power_w"], dtype=float)
            delta = float(np.max(np.abs(final - first)))
            maximum_power_change_w = max(maximum_power_change_w, delta)
            changed_power_groups += int(delta > 1e-15)
        step_delta = np.asarray(iterative_step_deltas, dtype=float)
        link_delta = np.asarray(iterative_link_abs_deltas, dtype=float)
        diagnostics = {
            "paired_steps": int(step_delta.size),
            "paired_links": int(link_delta.size),
            "unique_associations": int(len(iter_topology_cache)),
            "unique_cu_du_groups": int(len(iter_group_cache)),
            "groups_with_power_change_gt_1e_15_w": int(changed_power_groups),
            "maximum_group_power_change_w": float(maximum_power_change_w),
            "steps_with_nonzero_sum_rate_delta_gt_1e_12_gbps": int(np.sum(np.abs(step_delta) > 1e-12)),
            "links_with_nonzero_rate_delta_gt_1e_12_gbps": int(np.sum(link_delta > 1e-12)),
            "mean_iterative_minus_slapp_sum_rate_gbps": float(np.mean(step_delta)),
            "maximum_abs_step_sum_rate_delta_gbps": float(np.max(np.abs(step_delta))),
            "maximum_abs_link_rate_delta_gbps": float(np.max(link_delta)),
            "steps_with_any_soft_qos_indicator_change": int(iterative_qos_changed_steps),
            "shared_topology_metrics_by_design": [
                "joint_raw_success", "nrf_activity", "dus_moved"
            ],
        }
    return {
        "method": method, "train_seed": train_seed,
        "checkpoint": str(checkpoint), "episodes": episodes, "steps": steps,
        "summaries": summaries, "iterative_diagnostics": diagnostics,
    }


def fmt(mean: float, sd: float, scale: float = 1.0, digits: int = 3) -> str:
    return f"{scale * mean:.{digits}f} $\\pm$ {scale * sd:.{digits}f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model_results = []
    for method in ("EPA", "DPA2", "SLAPP"):
        for seed in range(1, 6):
            destination = OUT / f"{method.lower()}_trainseed{seed}.json"
            result = evaluate_model(method, seed)
            destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            model_results.append(result)
            print(f"completed {method} seed {seed}", flush=True)

    aggregate = {variant: {} for variant in VARIANT_ORDER}
    source_method = {"EPA": "EPA", "DPA2": "DPA2", "SLAPP": "SLAPP", "SINR_FP": "SLAPP", "SLAPP_ITER": "SLAPP"}
    for variant in VARIANT_ORDER:
        selected = [item for item in model_results if item["method"] == source_method[variant]]
        for metric in METRICS:
            values = np.array([item["summaries"][variant][metric] for item in selected])
            aggregate[variant][metric] = {
                "mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1)),
                "per_training_seed": values.tolist(),
            }
    payload = {
        "protocol": "50 common post-training episodes per frozen model; demands iid U[4,8] Gbps; 80 steps; no learning",
        "aggregation": "mean within each model, then mean and sample SD across five independent training seeds",
        "metrics": aggregate,
        "slapp_iter_diagnostics": [
            item["iterative_diagnostics"] for item in model_results
            if item["method"] == "SLAPP"
        ],
    }
    (OUT / "TABLE5_data.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # DPA2 is the retained internal PA=3/checkpoint identifier. The submitted
    # paper calls this method DPA, so manuscript-facing output must use DPA.
    labels = {"EPA": "EPA", "DPA2": "DPA", "SLAPP": "SLAPP", "SINR_FP": "SLAPP+SINR-PA", "SLAPP_ITER": "SLAPP-Iter"}
    rows = [
        ("Convergence score", "convergence_score", 1.0, 3, True),
        ("Sum rate (Gbps)", "sum_rate_gbps", 1.0, 2, True),
        ("QoS satisfaction (\\%)", "qos_satisfaction", 100.0, 2, True),
        ("Overall success rate (\\%)", "joint_raw_success", 100.0, 2, True),
        ("$N_{RF}$ projection activity", "nrf_activity", 1.0, 3, False),
        ("DU movement (avg.)", "dus_moved", 1.0, 3, False),
    ]
    lines = [
        "\\begin{table*}[!t]", "\\centering",
        "\\color{blue}",
        "\\caption{Post-training performance over 50 common evaluation episodes per fixed model under the training-demand distribution $d_j\\sim\\mathcal{U}[4,8]$~Gbps (80 steps per episode). Entries are mean $\\pm$ sample standard deviation across five independent training seeds. Success is timestep-level joint raw-action compliance as defined in~\\eqref{eq:overall_success}. SINR-PA and iterative SLAPP reuse the fixed SLAPP policies and identical projected topologies; no retraining is performed.}",
        "\\label{tab:final_training_metrics}",
        "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{lccccc}", "\\toprule",
        "\\textbf{Metric} & " + " & ".join(f"\\textbf{{{labels[v]}}}" for v in VARIANT_ORDER) + " \\\\",
        "\\midrule",
    ]
    for label, metric, scale, digits, higher in rows:
        means = [aggregate[v][metric]["mean"] for v in VARIANT_ORDER]
        best = (max(means) if higher else min(means))
        cells = []
        for variant, mean in zip(VARIANT_ORDER, means):
            cell = fmt(mean, aggregate[variant][metric]["sample_sd"], scale, digits)
            if abs(mean - best) <= 1e-12:
                cell = "\\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(label + " & " + " & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}", ""])
    (OUT / "TABLE5_posttraining_mean_sd.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"SAVED {OUT / 'TABLE5_posttraining_mean_sd.tex'}")


if __name__ == "__main__":
    main()
