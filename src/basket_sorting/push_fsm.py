from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PushFSMOutput:
    action: np.ndarray
    phase: str


class ScriptedPushFSM:
    """Perception-driven pushing expert for the final physics-based task."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.env_cfg = config["env"]
        self.fsm_cfg = config["fsm"]
        self.push_cfg = self.env_cfg.get("push", {})
        self.phase = "approach_push_start"
        self.counter = 0
        self.target_estimate: np.ndarray | None = None
        self.push_start: np.ndarray | None = None
        self.push_goal: np.ndarray | None = None

    def reset(self) -> None:
        self.phase = "approach_push_start"
        self.counter = 0
        self.target_estimate = None
        self.push_start = None
        self.push_goal = None

    def act(self, obs: dict) -> PushFSMOutput:
        task = obs["task"]
        ee = np.asarray(obs["ee_pos"], dtype=float)
        objects = obs.get("objects", {})
        baskets = obs.get("baskets", self.env_cfg["basket_poses"])
        if task.target_object in objects:
            self.target_estimate = np.asarray(objects[task.target_object], dtype=float)
        if self.target_estimate is None:
            self.target_estimate = ee.copy()

        basket = np.asarray(baskets[task.target_basket], dtype=float)
        push_z = float(self.push_cfg.get("push_z", self.env_cfg.get("grasp_z", 0.08)))
        safe_z = float(self.env_cfg["safe_z"])
        tol = float(self.fsm_cfg["tolerance"])
        speed_step = self._phase_step(task.speed_name)
        gripper = 1.0

        if self.push_start is None or self.push_goal is None:
            self._set_push_line(self.target_estimate, basket)

        assert self.push_start is not None
        assert self.push_goal is not None

        if self.phase == "approach_push_start":
            goal = np.array([self.push_start[0], self.push_start[1], safe_z], dtype=float)
            if self._near_xy(ee, goal, tol):
                self._advance("lower_to_push")
        elif self.phase == "lower_to_push":
            goal = np.array([self.push_start[0], self.push_start[1], push_z], dtype=float)
            if self._near(ee, goal, tol):
                self._advance("push")
        elif self.phase == "push":
            goal = np.array([self.push_goal[0], self.push_goal[1], push_z], dtype=float)
            self.counter += 1
            if self.counter >= int(self.push_cfg.get("push_steps", 90)):
                self._advance("retreat")
        elif self.phase == "retreat":
            goal = np.array([ee[0], ee[1], safe_z], dtype=float)
            self.counter += 1
            if self.counter >= int(self.fsm_cfg.get("retreat_steps", 8)):
                self._advance("done")
        else:
            goal = ee.copy()

        delta = self._clamped_delta(goal - ee, speed_step)
        return PushFSMOutput(action=np.array([delta[0], delta[1], delta[2], gripper], dtype=float), phase=self.phase)

    def _set_push_line(self, target: np.ndarray, basket: np.ndarray) -> None:
        target_xy = np.asarray(target[:2], dtype=float)
        basket_xy = np.asarray(basket[:2], dtype=float)
        direction = basket_xy - target_xy
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            direction = np.array([0.0, 1.0], dtype=float)
        else:
            direction = direction / norm
        start_offset = float(self.push_cfg.get("start_offset", 0.10))
        overshoot = float(self.push_cfg.get("goal_overshoot", 0.05))
        self.push_start = self._clip_xy(target_xy - direction * start_offset)
        self.push_goal = self._clip_xy(basket_xy + direction * overshoot)

    def _clip_xy(self, xy: np.ndarray) -> np.ndarray:
        bounds = self.env_cfg["workspace_bounds"]
        return np.array(
            [
                np.clip(xy[0], bounds["x"][0], bounds["x"][1]),
                np.clip(xy[1], bounds["y"][0], bounds["y"][1]),
            ],
            dtype=float,
        )

    def _phase_step(self, speed_name: str) -> float:
        return float(self.fsm_cfg["speed_to_step"].get(speed_name, self.fsm_cfg["speed_to_step"]["normal"]))

    def _advance(self, next_phase: str) -> None:
        self.phase = next_phase
        self.counter = 0

    def _near(self, a: np.ndarray, b: np.ndarray, tol: float) -> bool:
        return bool(np.linalg.norm(a - b) <= tol)

    def _near_xy(self, a: np.ndarray, b: np.ndarray, tol: float) -> bool:
        return bool(np.linalg.norm(a[:2] - b[:2]) <= tol)

    def _clamped_delta(self, delta: np.ndarray, max_step: float) -> np.ndarray:
        norm = float(np.linalg.norm(delta))
        if norm <= max_step or norm == 0.0:
            return delta
        return delta / norm * max_step
