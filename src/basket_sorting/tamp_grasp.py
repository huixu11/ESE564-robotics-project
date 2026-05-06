from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from basket_sorting.rrt import (
    RRTConnectConfig,
    RRTConnectPlanner,
    object_obstacles_from_config,
    workspace_bounds_from_config,
)


@dataclass(frozen=True)
class GraspCandidate:
    object_name: str
    grasp_pos: np.ndarray
    grasp_width: float
    score: float
    reason: str = "ok"


@dataclass(frozen=True)
class TAMPGraspPlan:
    candidate: GraspCandidate
    pre_grasp: np.ndarray
    grasp: np.ndarray
    lift: np.ndarray
    pre_place: np.ndarray
    place: np.ndarray
    retreat: np.ndarray
    approach_path: tuple[np.ndarray, ...]
    transfer_path: tuple[np.ndarray, ...]
    retreat_path: tuple[np.ndarray, ...]


@dataclass
class TAMPGraspOutput:
    action: np.ndarray
    phase: str


class ProjectTAMPGraspPlanner:
    """Small task-and-motion planner for the class Panda basket task.

    The symbolic task is always `place(object, basket)`. Continuous values are
    generated as grasp candidates and accepted only if they pass simple
    workspace, gripper-width, and clearance checks. This keeps the implementation
    project-sized while preserving the TAMP structure from the references in
    plan.md.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.env_cfg = config["env"]
        self.grasp_cfg = config.get("tamp_grasp", {})
        self.rejections: list[GraspCandidate] = []

    def plan(self, obs: dict[str, Any]) -> TAMPGraspPlan:
        task = obs["task"]
        target = np.asarray(obs["objects"][task.target_object], dtype=float)
        basket = np.asarray(obs.get("baskets", self.env_cfg["basket_poses"])[task.target_basket], dtype=float)
        self.rejections = []
        candidates = self._generate_candidates(task.target_object, target)
        feasible: list[GraspCandidate] = []
        for candidate in candidates:
            plan = self._make_plan(candidate, basket)
            reason = self._infeasible_reason(candidate, plan)
            if reason is None:
                feasible.append(candidate)
            else:
                self.rejections.append(
                    GraspCandidate(
                        object_name=candidate.object_name,
                        grasp_pos=candidate.grasp_pos,
                        grasp_width=candidate.grasp_width,
                        score=candidate.score,
                        reason=reason,
                    )
                )
        if not feasible:
            reasons = ", ".join(sorted({candidate.reason for candidate in self.rejections})) or "no candidates"
            raise RuntimeError(f"No feasible TAMP grasp plan for {task.instruction!r}: {reasons}")
        best = min(feasible, key=lambda candidate: candidate.score)
        return self._add_motion_paths(self._make_plan(best, basket), obs)

    def _generate_candidates(self, object_name: str, target: np.ndarray) -> list[GraspCandidate]:
        specs = self.grasp_cfg.get("object_specs", {})
        spec = specs.get(object_name, {})
        grasp_width = float(spec.get("grasp_width", self.grasp_cfg.get("default_grasp_width", 0.055)))
        base_z = float(spec.get("grasp_z", target[2]))
        z_offsets = self.grasp_cfg.get("candidate_z_offsets", [0.0])
        xy_offsets = self.grasp_cfg.get("candidate_xy_offsets", [[0.0, 0.0]])
        candidates: list[GraspCandidate] = []
        for xy_offset in xy_offsets:
            offset = np.asarray(xy_offset, dtype=float)
            if offset.shape != (2,):
                raise ValueError(f"candidate_xy_offsets entries must have shape (2,), got {offset.shape}")
            for z_offset in z_offsets:
                grasp_pos = np.array(
                    [target[0] + offset[0], target[1] + offset[1], base_z + float(z_offset)],
                    dtype=float,
                )
                score = float(np.linalg.norm(offset)) + abs(float(z_offset))
                candidates.append(
                    GraspCandidate(
                        object_name=object_name,
                        grasp_pos=grasp_pos,
                        grasp_width=grasp_width,
                        score=score,
                    )
                )
        return candidates

    def _make_plan(self, candidate: GraspCandidate, basket: np.ndarray) -> TAMPGraspPlan:
        pre_grasp_z = float(self.grasp_cfg.get("pre_grasp_z", self.env_cfg["safe_z"]))
        lift_z = float(self.grasp_cfg.get("lift_z", self.env_cfg["safe_z"]))
        place_z = float(self.grasp_cfg.get("place_z", self.env_cfg["place_z"]))
        retreat_z = float(self.grasp_cfg.get("retreat_z", lift_z))
        grasp = candidate.grasp_pos.copy()
        basket_xy = np.asarray(basket[:2], dtype=float)
        return TAMPGraspPlan(
            candidate=candidate,
            pre_grasp=np.array([grasp[0], grasp[1], pre_grasp_z], dtype=float),
            grasp=grasp,
            lift=np.array([grasp[0], grasp[1], lift_z], dtype=float),
            pre_place=np.array([basket_xy[0], basket_xy[1], lift_z], dtype=float),
            place=np.array([basket_xy[0], basket_xy[1], place_z], dtype=float),
            retreat=np.array([basket_xy[0], basket_xy[1], retreat_z], dtype=float),
            approach_path=(),
            transfer_path=(),
            retreat_path=(),
        )

    def _infeasible_reason(self, candidate: GraspCandidate, plan: TAMPGraspPlan) -> str | None:
        max_width = float(self.grasp_cfg.get("max_grasp_width", 0.080))
        if candidate.grasp_width > max_width:
            return "object wider than gripper"
        min_clearance = float(self.grasp_cfg.get("min_lift_clearance", 0.030))
        if plan.lift[2] - plan.grasp[2] < min_clearance:
            return "insufficient lift clearance"
        for name in ("pre_grasp", "grasp", "lift", "pre_place", "place", "retreat"):
            waypoint = getattr(plan, name)
            if not self._inside_workspace(waypoint):
                return f"{name} outside workspace"
        return None

    def _inside_workspace(self, pos: np.ndarray) -> bool:
        bounds = self.env_cfg["workspace_bounds"]
        return bool(
            bounds["x"][0] <= pos[0] <= bounds["x"][1]
            and bounds["y"][0] <= pos[1] <= bounds["y"][1]
            and bounds["z"][0] <= pos[2] <= bounds["z"][1]
        )

    def _add_motion_paths(self, plan: TAMPGraspPlan, obs: dict[str, Any]) -> TAMPGraspPlan:
        rrt_cfg = self.grasp_cfg.get("rrt", {})
        if not rrt_cfg.get("enabled", False):
            return self._replace_paths(plan, (plan.pre_grasp,), (plan.pre_place,), (plan.retreat,))

        objects = self._planning_objects(obs)
        start = np.asarray(obs.get("ee_pos", plan.pre_grasp), dtype=float)
        approach_path = self._plan_workspace_path("approach", start, plan.pre_grasp, objects)
        transfer_path = self._plan_workspace_path(
            "transfer",
            plan.lift,
            plan.pre_place,
            objects,
            ignored_objects=[plan.candidate.object_name],
        )
        retreat_path = self._plan_workspace_path("retreat", plan.place, plan.retreat, objects)
        return self._replace_paths(plan, tuple(approach_path), tuple(transfer_path), tuple(retreat_path))

    def _replace_paths(
        self,
        plan: TAMPGraspPlan,
        approach_path: tuple[np.ndarray, ...],
        transfer_path: tuple[np.ndarray, ...],
        retreat_path: tuple[np.ndarray, ...],
    ) -> TAMPGraspPlan:
        return TAMPGraspPlan(
            candidate=plan.candidate,
            pre_grasp=plan.pre_grasp,
            grasp=plan.grasp,
            lift=plan.lift,
            pre_place=plan.pre_place,
            place=plan.place,
            retreat=plan.retreat,
            approach_path=approach_path,
            transfer_path=transfer_path,
            retreat_path=retreat_path,
        )

    def _planning_objects(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        objects = {name: np.asarray(pos, dtype=float) for name, pos in obs.get("sim_objects", {}).items()}
        for name, pos in obs.get("objects", {}).items():
            objects[name] = np.asarray(pos, dtype=float)
        return objects

    def _plan_workspace_path(
        self,
        label: str,
        start: np.ndarray,
        goal: np.ndarray,
        objects: dict[str, np.ndarray],
        ignored_objects: list[str] | None = None,
    ) -> list[np.ndarray]:
        rrt_cfg = self.grasp_cfg.get("rrt", {})
        clearance = np.asarray(rrt_cfg.get("clearance", [0.025, 0.025, 0.030]), dtype=float)
        obstacles = object_obstacles_from_config(
            objects,
            rrt_cfg.get("object_half_extents", {}),
            ignored=ignored_objects or [],
        )
        config = RRTConnectConfig(
            bounds=workspace_bounds_from_config(self.env_cfg),
            step_size=float(rrt_cfg.get("step_size", 0.055)),
            goal_tolerance=float(rrt_cfg.get("goal_tolerance", 0.025)),
            max_iterations=int(rrt_cfg.get("max_iterations", 180)),
            line_resolution=float(rrt_cfg.get("line_resolution", 0.012)),
            clearance=clearance,
            smooth_iterations=int(rrt_cfg.get("smooth_iterations", 24)),
            goal_sample_rate=float(rrt_cfg.get("goal_sample_rate", 0.12)),
            seed=int(rrt_cfg.get("seed", 7)) + sum(ord(ch) for ch in label),
        )
        planner = RRTConnectPlanner(config, obstacles=obstacles)
        path = planner.plan(np.asarray(start, dtype=float), np.asarray(goal, dtype=float))
        if path is not None:
            return path
        return self._fallback_lifted_path(start, goal, planner)

    def _fallback_lifted_path(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        planner: RRTConnectPlanner,
    ) -> list[np.ndarray]:
        safe_z = min(
            planner.config.bounds[2, 1],
            max(float(self.grasp_cfg.get("rrt", {}).get("fallback_z", self.env_cfg["safe_z"])), start[2], goal[2]),
        )
        lifted_start = np.array([start[0], start[1], safe_z], dtype=float)
        lifted_goal = np.array([goal[0], goal[1], safe_z], dtype=float)
        path = [np.asarray(start, dtype=float), lifted_start, lifted_goal, np.asarray(goal, dtype=float)]
        if all(planner.checker.segment_is_valid(path[idx], path[idx + 1]) for idx in range(len(path) - 1)):
            return path
        return [np.asarray(start, dtype=float), np.asarray(goal, dtype=float)]


class ScriptedTAMPGraspPolicy:
    """Executes the selected TAMP grasp skeleton through the shared action API."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.env_cfg = config["env"]
        self.fsm_cfg = config["fsm"]
        self.grasp_cfg = config.get("tamp_grasp", {})
        self.planner = ProjectTAMPGraspPlanner(config)
        self.plan: TAMPGraspPlan | None = None
        self.phase = "tamp_plan"
        self.counter = 0
        self.path_indices: dict[str, int] = {}

    def reset(self) -> None:
        self.plan = None
        self.phase = "tamp_plan"
        self.counter = 0
        self.path_indices = {}

    def act(self, obs: dict[str, Any]) -> TAMPGraspOutput:
        if self.plan is None:
            self.plan = self.planner.plan(obs)
            self.phase = self._path_phase("rrt_approach", "pre_grasp")
            self.counter = 0

        assert self.plan is not None
        task = obs["task"]
        ee = np.asarray(obs["ee_pos"], dtype=float)
        tol = float(self.fsm_cfg["tolerance"])
        speed_step = self._phase_step(task.speed_name)
        gripper = 1.0

        goal: np.ndarray | None = None
        for _ in range(4):
            if self.phase in ("pre_grasp", "rrt_approach"):
                gripper = 1.0
                goal = self._path_goal(self.phase, self.plan.approach_path, "grasp", ee, tol)
                if goal is None:
                    continue
            elif self.phase in ("transfer", "rrt_transfer"):
                gripper = -1.0
                goal = self._path_goal(self.phase, self.plan.transfer_path, "place", ee, tol)
                if goal is None:
                    continue
            elif self.phase in ("retreat", "rrt_retreat"):
                gripper = 1.0
                goal = self._path_goal(self.phase, self.plan.retreat_path, "done", ee, tol)
                if goal is None:
                    continue
            break

        if goal is None and self.phase == "grasp":
            goal = self.plan.grasp
            gripper = 1.0
            if self._near(ee, goal, tol):
                self._advance("close")
        elif goal is None and self.phase == "close":
            goal = self.plan.grasp
            gripper = -1.0
            self.counter += 1
            if self.counter >= int(self.grasp_cfg.get("close_steps", self.fsm_cfg.get("grasp_hold_steps", 8))):
                self._advance("lift")
        elif goal is None and self.phase == "lift":
            goal = self.plan.lift
            gripper = -1.0
            if self._near(ee, goal, tol):
                self._advance(self._path_phase("rrt_transfer", "transfer"))
        elif goal is None and self.phase == "place":
            goal = self.plan.place
            gripper = -1.0
            if self._near(ee, goal, tol):
                self._advance("release")
        elif goal is None and self.phase == "release":
            goal = self.plan.place
            gripper = 1.0
            self.counter += 1
            if self.counter >= int(self.grasp_cfg.get("release_steps", self.fsm_cfg.get("release_hold_steps", 8))):
                self._advance(self._path_phase("rrt_retreat", "retreat"))
        elif goal is None:
            goal = ee.copy()
            gripper = 1.0

        delta = self._clamped_delta(goal - ee, speed_step)
        return TAMPGraspOutput(
            action=np.array([delta[0], delta[1], delta[2], gripper], dtype=float),
            phase=self.phase,
        )

    def _phase_step(self, speed_name: str) -> float:
        return float(self.fsm_cfg["speed_to_step"].get(speed_name, self.fsm_cfg["speed_to_step"]["normal"]))

    def _advance(self, next_phase: str) -> None:
        self.phase = next_phase
        self.counter = 0

    def _path_phase(self, rrt_phase: str, fallback_phase: str) -> str:
        return rrt_phase if self.grasp_cfg.get("rrt", {}).get("enabled", False) else fallback_phase

    def _path_goal(
        self,
        phase: str,
        path: tuple[np.ndarray, ...],
        next_phase: str,
        ee: np.ndarray,
        tol: float,
    ) -> np.ndarray | None:
        if not path:
            self._advance(next_phase)
            return None
        index = self.path_indices.get(phase, 0)
        while index < len(path) and self._near(ee, path[index], tol):
            index += 1
        if index >= len(path):
            self.path_indices[phase] = 0
            self._advance(next_phase)
            return None
        self.path_indices[phase] = index
        return path[index]

    def _near(self, a: np.ndarray, b: np.ndarray, tol: float) -> bool:
        return bool(np.linalg.norm(a - b) <= tol)

    def _clamped_delta(self, delta: np.ndarray, max_step: float) -> np.ndarray:
        norm = float(np.linalg.norm(delta))
        if norm <= max_step or norm == 0.0:
            return delta
        return delta / norm * max_step
