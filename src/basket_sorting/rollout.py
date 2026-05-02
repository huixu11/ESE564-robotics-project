from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


class Policy(Protocol):
    def reset(self) -> None:
        ...

    def act(self, obs: dict[str, Any]):
        ...


@dataclass
class RolloutResult:
    success: bool
    steps: int
    total_reward: float
    frames: list[np.ndarray] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)
    instruction: str = ""


def run_episode(
    env,
    policy: Policy,
    instruction: str | None = None,
    seed: int | None = None,
    record: bool = False,
    frames: bool = False,
) -> RolloutResult:
    obs = env.reset(instruction=instruction, seed=seed)
    policy.reset()
    total_reward = 0.0
    result_frames: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    done = False
    info = {"success": False, "steps": 0}

    while not done:
        output = policy.act(obs)
        action = output.action if hasattr(output, "action") else np.asarray(output, dtype=float)
        phase = output.phase if hasattr(output, "phase") else obs.get("phase", "policy")
        if hasattr(env, "phase"):
            env.phase = phase
        if frames:
            result_frames.append(obs["rgb"])
        if record:
            records.append(
                {
                    "rgb": obs["rgb"],
                    "state_features": obs["state_features"],
                    "subtask_text": phase,
                    "speed": obs["task"].speed,
                    "action": action,
                    "phase": phase,
                }
            )
        obs, reward, done, info = env.step(action)
        total_reward += reward

    if frames:
        result_frames.append(obs["rgb"])
    return RolloutResult(
        success=bool(info["success"]),
        steps=int(info["steps"]),
        total_reward=float(total_reward),
        frames=result_frames,
        records=records,
        instruction=obs["task"].instruction,
    )
