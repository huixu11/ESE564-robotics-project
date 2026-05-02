from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PHASES = (
    "approach",
    "pre_grasp",
    "grasp",
    "lift",
    "transfer",
    "place",
    "release",
    "retreat",
    "done",
)


@dataclass
class FSMOutput:
    action: np.ndarray
    phase: str


class ScriptedPickPlaceFSM:
    """Scripted expert that outputs the shared low-level action interface."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.env_cfg = config["env"]
        self.fsm_cfg = config["fsm"]
        self.phase = "approach"
        self.counter = 0

    def reset(self) -> None:
        self.phase = "approach"
        self.counter = 0

    def act(self, obs: dict) -> FSMOutput:
        task = obs["task"]
        ee = obs["ee_pos"]
        target = obs["objects"][task.target_object]
        basket = np.asarray(self.env_cfg["basket_poses"][task.target_basket], dtype=float)
        tol = float(self.fsm_cfg["tolerance"])

        speed_step = self._phase_step(task.speed_name)
        gripper = 1.0

        if self.phase == "approach":
            goal = np.array([target[0], target[1], self.env_cfg["safe_z"]], dtype=float)
            gripper = 1.0
            if self._near(ee, goal, tol):
                self._advance("pre_grasp")
        elif self.phase == "pre_grasp":
            goal = np.array([target[0], target[1], self.env_cfg["grasp_z"]], dtype=float)
            gripper = 1.0
            if self._near(ee, goal, tol):
                self._advance("grasp")
        elif self.phase == "grasp":
            goal = ee.copy()
            gripper = -1.0
            self.counter += 1
            if obs["attached_object"] == task.target_object or self.counter >= int(self.fsm_cfg["grasp_hold_steps"]):
                self._advance("lift")
        elif self.phase == "lift":
            goal = np.array([ee[0], ee[1], self.env_cfg["safe_z"]], dtype=float)
            gripper = -1.0
            if self._near(ee, goal, tol):
                self._advance("transfer")
        elif self.phase == "transfer":
            goal = np.array([basket[0], basket[1], self.env_cfg["safe_z"]], dtype=float)
            gripper = -1.0
            if self._near(ee, goal, tol):
                self._advance("place")
        elif self.phase == "place":
            goal = np.array([basket[0], basket[1], self.env_cfg["place_z"]], dtype=float)
            gripper = -1.0
            if self._near(ee, goal, tol):
                self._advance("release")
        elif self.phase == "release":
            goal = ee.copy()
            gripper = 1.0
            self.counter += 1
            if self.counter >= int(self.fsm_cfg["release_hold_steps"]):
                self._advance("retreat")
        elif self.phase == "retreat":
            goal = np.array([ee[0], ee[1], self.env_cfg["safe_z"]], dtype=float)
            gripper = 1.0
            self.counter += 1
            if self.counter >= int(self.fsm_cfg["retreat_steps"]):
                self._advance("done")
        else:
            goal = ee.copy()
            gripper = 1.0

        delta = self._clamped_delta(goal - ee, speed_step)
        return FSMOutput(action=np.array([delta[0], delta[1], delta[2], gripper], dtype=float), phase=self.phase)

    def _phase_step(self, speed_name: str) -> float:
        return float(self.fsm_cfg["speed_to_step"].get(speed_name, self.fsm_cfg["speed_to_step"]["normal"]))

    def _advance(self, next_phase: str) -> None:
        self.phase = next_phase
        self.counter = 0

    def _near(self, a: np.ndarray, b: np.ndarray, tol: float) -> bool:
        return bool(np.linalg.norm(a - b) <= tol)

    def _clamped_delta(self, delta: np.ndarray, max_step: float) -> np.ndarray:
        norm = float(np.linalg.norm(delta))
        if norm <= max_step or norm == 0.0:
            return delta
        return delta / norm * max_step
