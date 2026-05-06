from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class AABBObstacle:
    name: str
    center: np.ndarray
    half_extents: np.ndarray

    def contains(self, point: np.ndarray, clearance: np.ndarray) -> bool:
        delta = np.abs(np.asarray(point, dtype=float) - self.center)
        return bool(np.all(delta <= self.half_extents + clearance))


@dataclass(frozen=True)
class RRTConnectConfig:
    bounds: np.ndarray
    step_size: float = 0.055
    goal_tolerance: float = 0.025
    max_iterations: int = 180
    line_resolution: float = 0.012
    clearance: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    smooth_iterations: int = 24
    goal_sample_rate: float = 0.12
    seed: int = 7


class WorkspaceCollisionChecker:
    def __init__(
        self,
        bounds: np.ndarray,
        obstacles: Iterable[AABBObstacle],
        clearance: np.ndarray,
        line_resolution: float,
    ) -> None:
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.shape != (3, 2):
            raise ValueError(f"bounds must have shape (3, 2), got {self.bounds.shape}")
        self.obstacles = list(obstacles)
        self.clearance = np.asarray(clearance, dtype=float)
        if self.clearance.shape != (3,):
            raise ValueError(f"clearance must have shape (3,), got {self.clearance.shape}")
        self.line_resolution = float(line_resolution)

    def point_is_valid(self, point: np.ndarray) -> bool:
        point = np.asarray(point, dtype=float)
        if point.shape != (3,):
            raise ValueError(f"point must have shape (3,), got {point.shape}")
        if np.any(point < self.bounds[:, 0]) or np.any(point > self.bounds[:, 1]):
            return False
        return not any(obstacle.contains(point, self.clearance) for obstacle in self.obstacles)

    def segment_is_valid(self, start: np.ndarray, goal: np.ndarray) -> bool:
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        distance = float(np.linalg.norm(goal - start))
        if distance == 0.0:
            return self.point_is_valid(start)
        steps = max(1, int(np.ceil(distance / self.line_resolution)))
        for idx in range(steps + 1):
            alpha = float(idx) / float(steps)
            if not self.point_is_valid(start * (1.0 - alpha) + goal * alpha):
                return False
        return True


class _Tree:
    def __init__(self, root: np.ndarray) -> None:
        self.nodes = [np.asarray(root, dtype=float)]
        self.parents = [-1]

    def nearest(self, point: np.ndarray) -> int:
        distances = [float(np.linalg.norm(node - point)) for node in self.nodes]
        return int(np.argmin(distances))

    def add(self, point: np.ndarray, parent: int) -> int:
        self.nodes.append(np.asarray(point, dtype=float))
        self.parents.append(parent)
        return len(self.nodes) - 1

    def path_to_root(self, index: int) -> list[np.ndarray]:
        path: list[np.ndarray] = []
        while index >= 0:
            path.append(self.nodes[index])
            index = self.parents[index]
        path.reverse()
        return path


class RRTConnectPlanner:
    """Bidirectional RRT-Connect for short workspace paths.

    The project controller accepts Cartesian end-effector deltas. This planner
    therefore searches in 3D workspace coordinates, using inflated object boxes as
    a planning-scene approximation before differential IK executes the result.
    """

    def __init__(self, config: RRTConnectConfig, obstacles: Iterable[AABBObstacle] = ()) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.checker = WorkspaceCollisionChecker(
            bounds=config.bounds,
            obstacles=obstacles,
            clearance=config.clearance,
            line_resolution=config.line_resolution,
        )

    def plan(self, start: np.ndarray, goal: np.ndarray) -> list[np.ndarray] | None:
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        if start.shape != (3,) or goal.shape != (3,):
            raise ValueError(f"start and goal must both have shape (3,), got {start.shape} and {goal.shape}")
        if not self.checker.point_is_valid(start) or not self.checker.point_is_valid(goal):
            return None
        if self.checker.segment_is_valid(start, goal):
            return [start, goal]

        start_tree = _Tree(start)
        goal_tree = _Tree(goal)
        for _ in range(int(self.config.max_iterations)):
            sample = self._sample(goal)
            start_status, start_idx = self._extend(start_tree, sample)
            if start_status != "trapped":
                connect_status, goal_idx = self._connect(goal_tree, start_tree.nodes[start_idx])
                if connect_status == "reached":
                    return self._smooth(self._combine(start_tree, start_idx, goal_tree, goal_idx))

            sample = self._sample(start)
            goal_status, goal_idx = self._extend(goal_tree, sample)
            if goal_status != "trapped":
                connect_status, start_idx = self._connect(start_tree, goal_tree.nodes[goal_idx])
                if connect_status == "reached":
                    return self._smooth(self._combine(start_tree, start_idx, goal_tree, goal_idx))

        return None

    def _sample(self, bias: np.ndarray) -> np.ndarray:
        if self.rng.random() < float(self.config.goal_sample_rate):
            return np.asarray(bias, dtype=float)
        return self.rng.uniform(self.config.bounds[:, 0], self.config.bounds[:, 1])

    def _steer(self, start: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, bool]:
        delta = target - start
        distance = float(np.linalg.norm(delta))
        if distance <= self.config.step_size:
            return target, True
        return start + delta / distance * self.config.step_size, False

    def _extend(self, tree: _Tree, target: np.ndarray) -> tuple[str, int]:
        nearest_idx = tree.nearest(target)
        nearest = tree.nodes[nearest_idx]
        new_point, reached = self._steer(nearest, target)
        if not self.checker.point_is_valid(new_point):
            return "trapped", nearest_idx
        if not self.checker.segment_is_valid(nearest, new_point):
            return "trapped", nearest_idx
        new_idx = tree.add(new_point, nearest_idx)
        if reached:
            return "reached", new_idx
        if (
            np.linalg.norm(new_point - target) <= self.config.goal_tolerance
            and self.checker.point_is_valid(target)
            and self.checker.segment_is_valid(new_point, target)
        ):
            target_idx = tree.add(target, new_idx)
            return "reached", target_idx
        return "advanced", new_idx

    def _connect(self, tree: _Tree, target: np.ndarray) -> tuple[str, int]:
        status = "advanced"
        index = tree.nearest(target)
        while status == "advanced":
            status, index = self._extend(tree, target)
        return status, index

    def _combine(self, start_tree: _Tree, start_idx: int, goal_tree: _Tree, goal_idx: int) -> list[np.ndarray]:
        start_path = start_tree.path_to_root(start_idx)
        goal_path = goal_tree.path_to_root(goal_idx)
        reverse_goal = list(reversed(goal_path))
        if np.allclose(start_path[-1], reverse_goal[0]):
            return start_path + reverse_goal[1:]
        return start_path + reverse_goal

    def _smooth(self, path: list[np.ndarray]) -> list[np.ndarray]:
        if len(path) <= 2:
            return path
        smoothed = [point.copy() for point in path]
        for _ in range(int(self.config.smooth_iterations)):
            if len(smoothed) <= 2:
                break
            i, j = sorted(self.rng.choice(len(smoothed), size=2, replace=False).tolist())
            if j <= i + 1:
                continue
            if self.checker.segment_is_valid(smoothed[i], smoothed[j]):
                smoothed = smoothed[: i + 1] + smoothed[j:]
        return smoothed


def workspace_bounds_from_config(env_cfg: dict) -> np.ndarray:
    bounds = env_cfg["workspace_bounds"]
    return np.array(
        [
            [bounds["x"][0], bounds["x"][1]],
            [bounds["y"][0], bounds["y"][1]],
            [bounds["z"][0], bounds["z"][1]],
        ],
        dtype=float,
    )


def object_obstacles_from_config(
    objects: dict[str, np.ndarray],
    object_half_extents: dict[str, list[float]],
    ignored: Iterable[str] = (),
) -> list[AABBObstacle]:
    ignored_names = set(ignored)
    obstacles: list[AABBObstacle] = []
    for name, pos in objects.items():
        if name in ignored_names or name not in object_half_extents:
            continue
        half_extents = np.asarray(object_half_extents[name], dtype=float)
        if half_extents.shape != (3,):
            raise ValueError(f"object_half_extents[{name!r}] must have shape (3,), got {half_extents.shape}")
        obstacles.append(
            AABBObstacle(
                name=name,
                center=np.asarray(pos, dtype=float),
                half_extents=half_extents,
            )
        )
    return obstacles
