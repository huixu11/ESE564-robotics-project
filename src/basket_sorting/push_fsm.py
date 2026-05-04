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
        self.segment_index = 0
        self.lateral_sign = 0.0
        self.target_estimate: np.ndarray | None = None
        self.push_start: np.ndarray | None = None
        self.push_goal: np.ndarray | None = None
        self.segment_object_goal: np.ndarray | None = None
        self.forced_target_estimate: np.ndarray | None = None

    def reset(self) -> None:
        self.phase = "approach_push_start"
        self.counter = 0
        self.segment_index = 0
        self.lateral_sign = 0.0
        self.target_estimate = None
        self.push_start = None
        self.push_goal = None
        self.segment_object_goal = None
        self.forced_target_estimate = None

    def act(self, obs: dict) -> PushFSMOutput:
        task = obs["task"]
        ee = np.asarray(obs["ee_pos"], dtype=float)
        objects = obs.get("objects", {})
        baskets = obs.get("baskets", self.env_cfg["basket_poses"])
        if (
            self.forced_target_estimate is not None
            and self.push_start is None
            and not self.phase.startswith("settle_after")
        ):
            self.target_estimate = self.forced_target_estimate.copy()
            self.forced_target_estimate = None
        elif task.target_object in objects:
            self.target_estimate = np.asarray(objects[task.target_object], dtype=float)
        if self.target_estimate is None:
            self.target_estimate = ee.copy()

        basket = np.asarray(baskets[task.target_basket], dtype=float)
        push_z = float(self.push_cfg.get("push_z", self.env_cfg.get("grasp_z", 0.08)))
        safe_z = float(self.env_cfg["safe_z"])
        tol = float(self.fsm_cfg["tolerance"])
        speed_step = self._phase_step(task.speed_name)
        gripper = 1.0

        if (self.push_start is None or self.push_goal is None) and not self.phase.startswith("settle_after"):
            self._set_push_line(self.target_estimate, basket)

        settle_steps = int(self.push_cfg.get("settle_steps", 0))

        if self.phase == "approach_push_start":
            assert self.push_start is not None
            goal = np.array([self.push_start[0], self.push_start[1], safe_z], dtype=float)
            if self._near_xy(ee, goal, tol):
                self._advance("lower_to_push")
        elif self.phase == "lower_to_push":
            assert self.push_start is not None
            goal = np.array([self.push_start[0], self.push_start[1], push_z], dtype=float)
            if self._near(ee, goal, tol):
                self._advance("settle_before_push" if settle_steps > 0 else "push")
        elif self.phase == "settle_before_push":
            goal = ee.copy()
            self.counter += 1
            if self.counter >= settle_steps:
                self._advance("push")
        elif self.phase == "push":
            assert self.push_goal is not None
            goal = np.array([self.push_goal[0], self.push_goal[1], push_z], dtype=float)
            self.counter += 1
            if self.counter >= int(self.push_cfg.get("push_steps", 90)):
                if self._advance_segment():
                    self._advance("settle_after_segment" if settle_steps > 0 else "approach_push_start")
                else:
                    self._advance("settle_after_push" if settle_steps > 0 else "retreat")
        elif self.phase == "settle_after_segment":
            goal = np.array([ee[0], ee[1], safe_z], dtype=float)
            self.counter += 1
            lift_steps = int(self.push_cfg.get("between_segment_lift_steps", max(settle_steps, 1)))
            if self._near(ee, goal, tol) or self.counter >= lift_steps:
                self.push_start = None
                self.push_goal = None
                self._advance("approach_push_start")
        elif self.phase == "settle_after_push":
            goal = ee.copy()
            self.counter += 1
            if self.counter >= settle_steps:
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
        if self.push_cfg.get("two_stage", True):
            direction, anchor_xy, goal_xy = self._axis_aligned_segment(target_xy, basket_xy)
        else:
            direction = basket_xy - target_xy
            norm = float(np.linalg.norm(direction))
            if norm < 1e-9:
                direction = np.array([0.0, 1.0], dtype=float)
            else:
                direction = direction / norm
            anchor_xy = target_xy
            goal_xy = basket_xy + direction * float(self.push_cfg.get("goal_overshoot", 0.05))
        start_offset = float(self.push_cfg.get("start_offset", 0.10))
        self.push_start = self._clip_xy(anchor_xy - direction * start_offset)
        goal_offset = float(self.push_cfg.get("contact_goal_offset", 0.0))
        self.push_goal = self._clip_xy(goal_xy - direction * goal_offset)
        lead_offset = float(self.push_cfg.get("predicted_lateral_lead_offset", 0.0))
        if self.segment_index == 0 and lead_offset > 0.0:
            self.segment_object_goal = self._clip_xy(self.push_goal + direction * lead_offset)
        else:
            self.segment_object_goal = self._clip_xy(goal_xy)

    def _axis_aligned_segment(
        self,
        target_xy: np.ndarray,
        basket_xy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        overshoot = float(self.push_cfg.get("goal_overshoot", 0.05))
        x_tolerance = float(self.push_cfg.get("x_alignment_tolerance", 0.04))
        if self.segment_index == 0 and abs(float(basket_xy[0] - target_xy[0])) > x_tolerance:
            sign = 1.0 if basket_xy[0] >= target_xy[0] else -1.0
            self.lateral_sign = sign
            direction = np.array([sign, 0.0], dtype=float)
            goal = np.array([basket_xy[0] + sign * overshoot, target_xy[1]], dtype=float)
            return direction, target_xy, goal

        self.segment_index = 1
        sign = 1.0 if basket_xy[1] >= target_xy[1] else -1.0
        direction = np.array([0.0, sign], dtype=float)
        half_extent = float(self.env_cfg.get("basket_half_extent", 0.105))
        if self.push_cfg.get("forward_uses_target_x", False):
            aligned_x = target_xy[0]
        elif self.lateral_sign != 0.0:
            aligned_x = basket_xy[0] - self.lateral_sign * half_extent * 0.8
        else:
            aligned_x = target_xy[0]
        anchor = np.array([aligned_x, target_xy[1]], dtype=float)
        goal = np.array([aligned_x, basket_xy[1] + sign * overshoot], dtype=float)
        return direction, anchor, goal

    def _advance_segment(self) -> bool:
        if not self.push_cfg.get("two_stage", True) or self.segment_index >= 1:
            return False
        self.segment_index = 1
        if self.push_cfg.get("use_predicted_segment_goal", False) and self.segment_object_goal is not None:
            self.forced_target_estimate = np.array(
                [self.segment_object_goal[0], self.segment_object_goal[1], self.target_estimate[2]],
                dtype=float,
            )
        self.push_start = None
        self.push_goal = None
        self.segment_object_goal = None
        return True

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
