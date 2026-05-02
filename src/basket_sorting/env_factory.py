from __future__ import annotations

from typing import Any

from basket_sorting.envs import KinematicBasketSortingEnv, MujocoBasketSortingEnv


def make_env(config: dict[str, Any], seed: int | None = None):
    env_name = config["env"].get("name", "kinematic")
    if env_name == "kinematic":
        return KinematicBasketSortingEnv(config, seed=seed)
    if env_name == "mujoco":
        return MujocoBasketSortingEnv(config, seed=seed)
    raise ValueError(f"Unknown env.name={env_name!r}")
