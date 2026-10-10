"""Episode-boundary full-state checkpointing for exact CHARM continuation."""
from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _buffer_state(buffer: Any) -> dict[str, Any]:
    return {
        "actions": copy.deepcopy(buffer.actions),
        "states": copy.deepcopy(buffer.states),
        "logprobs": copy.deepcopy(buffer.logprobs),
        "rewards": copy.deepcopy(buffer.rewards),
        "state_values": copy.deepcopy(buffer.state_values),
        "is_terminals": copy.deepcopy(buffer.is_terminals),
        "infos": copy.deepcopy(buffer.infos),
    }


def _restore_buffer(buffer: Any, state: dict[str, Any]) -> None:
    buffer.clear()
    for name, values in state.items():
        setattr(buffer, name, values)


def _rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_full_checkpoint(
    path: str | Path,
    *,
    agent: Any,
    env: Any,
    next_episode: int,
    timestep: int,
    histories: dict[str, Any],
    best_state: dict[str, Any],
    run_config: dict[str, Any],
) -> None:
    """Atomically save everything required at the next episode boundary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "boundary": "after_episode_housekeeping_before_next_reset",
        "next_episode": int(next_episode),
        "timestep": int(timestep),
        "policy_state_dict": agent.policy.state_dict(),
        "policy_old_state_dict": agent.policy_old.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "scheduler_state_dict": agent.scheduler.state_dict(),
        "agent_state": {
            "training_epoch": agent.training_epoch,
            "lambda_2": agent.lambda_2,
            "lambda_3": agent.lambda_3,
            "lambda_rate": agent.lambda_rate,
            "lambda_nrf": agent.lambda_nrf,
            "mu_rate": agent.mu_rate,
            "mu_nrf": agent.mu_nrf,
            "beta_deploy": agent.beta_deploy,
            "beta_assoc": agent.beta_assoc,
            "entropy_weight": agent.entropy_weight,
            "temp": agent.temp,
            "action_std": agent.action_std,
        },
        "rollout_buffer": _buffer_state(agent.buffer),
        "ppg_buffer": _buffer_state(agent.ppg_buffer),
        "environment_state": {
            "lambda_2": env.lambda_2,
            "lambda_3": env.lambda_3,
            "penalty_multipliers": copy.deepcopy(env.penalty_multipliers),
            "current_demands": env.current_demands.copy(),
            "prev_demands": env.prev_demands.copy(),
            "prev_rates": env.prev_rates.copy(),
            "rate_list": env.rate_list.copy(),
            "timestep": env.timestep,
            "episode_count": env.episode_count,
        },
        "histories": copy.deepcopy(histories),
        "best_state": copy.deepcopy(best_state),
        "rng_state": _rng_state(),
        "run_config": copy.deepcopy(run_config),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary, _use_new_zipfile_serialization=False)
    os.replace(temporary, path)


def load_full_checkpoint(
    path: str | Path,
    *,
    agent: Any,
    env: Any,
) -> dict[str, Any]:
    """Restore a full checkpoint and return counters/histories to the trainer."""
    payload = torch.load(path, map_location=agent.device, weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError(f"Unsupported full-checkpoint version: {payload.get('format_version')}")
    agent.policy.load_state_dict(payload["policy_state_dict"])
    agent.policy_old.load_state_dict(payload["policy_old_state_dict"])
    agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
    agent.scheduler.load_state_dict(payload["scheduler_state_dict"])
    for name, value in payload["agent_state"].items():
        setattr(agent, name, value)
    agent.policy.set_action_std(agent.action_std)
    agent.policy_old.set_action_std(agent.action_std)
    agent.policy.temp = agent.temp
    agent.policy_old.temp = agent.temp
    _restore_buffer(agent.buffer, payload["rollout_buffer"])
    _restore_buffer(agent.ppg_buffer, payload["ppg_buffer"])

    for name, value in payload["environment_state"].items():
        setattr(env, name, value)
    _restore_rng(payload["rng_state"])
    return payload
