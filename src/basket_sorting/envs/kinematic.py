from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from basket_sorting.controllers import DifferentialIKController, ToyPandaKinematics
from basket_sorting.tasks import TaskSpec, make_instruction, parse_instruction


@dataclass
class ObjectState:
    name: str
    pos: np.ndarray
    held: bool = False


class KinematicBasketSortingEnv:
    """Runnably approximates the MuJoCo task while preserving final interfaces."""

    def __init__(self, config: dict[str, Any], seed: int | None = None) -> None:
        self.config = config
        self.env_cfg = config["env"]
        self.rng = np.random.default_rng(config.get("seed", 0) if seed is None else seed)
        self.kinematics = ToyPandaKinematics()
        self.controller = DifferentialIKController(
            kinematics=self.kinematics,
            damping=config["controller"]["damping"],
            max_ee_step=config["controller"]["max_ee_step"],
            max_joint_step=config["controller"]["max_joint_step"],
            max_joint_jump=config["controller"]["max_joint_jump"],
            joint_limits=config["controller"]["joint_limits"],
        )
        self.qpos = np.array([0.0, -0.12, self.env_cfg["safe_z"], 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.ee_pos = self.kinematics.forward(self.qpos)
        self.gripper_open = True
        self.attached_object: str | None = None
        self.objects: dict[str, ObjectState] = {}
        self.task: TaskSpec | None = None
        self.steps = 0
        self.last_ik_success = True
        self.phase = "reset"

    def reset(self, instruction: str | None = None, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        object_names = list(self.env_cfg["object_names"])
        target_object = str(self.rng.choice(object_names))
        distractor = object_names[1 - object_names.index(target_object)]
        target_basket = str(self.rng.choice(list(self.env_cfg["basket_poses"].keys())))
        speed_name = str(self.rng.choice(["careful", "normal", "fast"], p=[0.2, 0.6, 0.2]))
        instruction = instruction or make_instruction(target_object, target_basket, speed_name)
        self.task = parse_instruction(instruction)

        self.objects = {
            self.task.target_object: ObjectState(self.task.target_object, self._sample_object_pos()),
            distractor: ObjectState(distractor, self._sample_object_pos()),
        }
        while np.linalg.norm(self.objects[self.task.target_object].pos[:2] - self.objects[distractor].pos[:2]) < 0.15:
            self.objects[distractor].pos = self._sample_object_pos()

        self.qpos = np.array([0.0, -0.18, self.env_cfg["safe_z"], 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.ee_pos = self.kinematics.forward(self.qpos)
        self.gripper_open = True
        self.attached_object = None
        self.steps = 0
        self.last_ik_success = True
        self.phase = "reset"
        return self._observation()

    def step(self, action: np.ndarray | list[float]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        action_arr = np.asarray(action, dtype=float)
        if action_arr.shape != (4,):
            raise ValueError(f"action must have shape (4,), got {action_arr.shape}")

        ik = self.controller.solve(self.qpos, action_arr[:3])
        self.last_ik_success = ik.success
        if ik.success:
            self.qpos = ik.qpos
            self.ee_pos = self.kinematics.forward(self.qpos)
        self.ee_pos = self._clip_workspace(self.ee_pos)
        self.qpos[:3] = self.ee_pos

        self.gripper_open = bool(action_arr[3] >= 0.0)
        self._update_attachment()
        self.steps += 1

        success = self.is_success()
        done = success or self.steps >= int(self.env_cfg["max_steps"])
        reward = 1.0 if success else -0.001
        info = {
            "success": success,
            "steps": self.steps,
            "ik_success": self.last_ik_success,
            "phase": self.phase,
        }
        return self._observation(), reward, done, info

    def render_rgb(self) -> np.ndarray:
        width = int(self.env_cfg["camera"]["width"])
        height = int(self.env_cfg["camera"]["height"])
        image = Image.new("RGB", (width, height), (236, 232, 220))
        draw = ImageDraw.Draw(image)

        for name, center in self.env_cfg["basket_poses"].items():
            xy = np.asarray(center[:2], dtype=float)
            half = float(self.env_cfg["basket_half_extent"])
            p0 = self._world_to_pixel(xy - half, width, height)
            p1 = self._world_to_pixel(xy + half, width, height)
            box = [
                (min(p0[0], p1[0]), min(p0[1], p1[1])),
                (max(p0[0], p1[0]), max(p0[1], p1[1])),
            ]
            color = (64, 130, 200) if name == "left" else (80, 165, 110)
            draw.rectangle(box, outline=color, width=3)

        colors = {
            "cracker_box": (205, 80, 60),
            "mustard_bottle": (225, 190, 55),
        }
        for obj in self.objects.values():
            px, py = self._world_to_pixel(obj.pos[:2], width, height)
            radius = 7
            draw.ellipse(
                [(px - radius, py - radius), (px + radius, py + radius)],
                fill=colors.get(obj.name, (130, 130, 130)),
                outline=(40, 40, 40),
            )

        ex, ey = self._world_to_pixel(self.ee_pos[:2], width, height)
        draw.line([(ex - 8, ey), (ex + 8, ey)], fill=(30, 30, 30), width=2)
        draw.line([(ex, ey - 8), (ex, ey + 8)], fill=(30, 30, 30), width=2)
        return np.asarray(image)

    def save_frame(self, path: str | Path) -> None:
        Image.fromarray(self.render_rgb()).save(path)

    def is_success(self) -> bool:
        if self.task is None:
            return False
        obj = self.objects[self.task.target_object]
        basket_center = np.asarray(self.env_cfg["basket_poses"][self.task.target_basket][:2], dtype=float)
        half = float(self.env_cfg["basket_half_extent"])
        inside = np.all(np.abs(obj.pos[:2] - basket_center) <= half)
        return bool(inside and self.attached_object is None and obj.pos[2] <= 0.025)

    def get_state_features(self) -> np.ndarray:
        if self.task is None:
            raise RuntimeError("Call reset before requesting features.")
        target = self.objects[self.task.target_object].pos
        basket = np.asarray(self.env_cfg["basket_poses"][self.task.target_basket], dtype=float)
        attached = 1.0 if self.attached_object == self.task.target_object else 0.0
        return np.concatenate(
            [
                self.ee_pos,
                target,
                basket,
                target - self.ee_pos,
                basket - self.ee_pos,
                np.array(
                    [
                        float(self.env_cfg["safe_z"]) - self.ee_pos[2],
                        float(self.env_cfg["grasp_z"]) - self.ee_pos[2],
                        float(self.env_cfg["place_z"]) - self.ee_pos[2],
                        1.0 if self.gripper_open else 0.0,
                        self.task.speed,
                        attached,
                    ],
                    dtype=float,
                ),
            ]
        )

    def _sample_object_pos(self) -> np.ndarray:
        bounds = self.env_cfg["table_bounds"]
        x = self.rng.uniform(*bounds["x"])
        y = self.rng.uniform(-0.22, 0.12)
        return np.array([x, y, 0.0], dtype=float)

    def _update_attachment(self) -> None:
        if self.task is None:
            return
        target = self.objects[self.task.target_object]
        if self.gripper_open:
            if self.attached_object is not None:
                dropped = self.objects[self.attached_object]
                dropped.held = False
                dropped.pos = np.array([self.ee_pos[0], self.ee_pos[1], 0.0], dtype=float)
                self.attached_object = None
            return

        if self.attached_object is None:
            xy_dist = np.linalg.norm(target.pos[:2] - self.ee_pos[:2])
            z_dist = abs(float(self.ee_pos[2]) - float(self.env_cfg["grasp_z"]))
            if xy_dist <= float(self.env_cfg["grasp_radius"]) and z_dist <= 0.045:
                self.attached_object = target.name
                target.held = True

        if self.attached_object is not None:
            held = self.objects[self.attached_object]
            held.pos = np.array([self.ee_pos[0], self.ee_pos[1], max(0.0, self.ee_pos[2] - 0.04)], dtype=float)

    def _observation(self) -> dict[str, Any]:
        if self.task is None:
            raise RuntimeError("Environment has not been reset.")
        return {
            "rgb": self.render_rgb(),
            "qpos": self.qpos.copy(),
            "ee_pos": self.ee_pos.copy(),
            "gripper_open": self.gripper_open,
            "task": self.task,
            "objects": {name: obj.pos.copy() for name, obj in self.objects.items()},
            "attached_object": self.attached_object,
            "state_features": self.get_state_features(),
            "phase": self.phase,
        }

    def _clip_workspace(self, pos: np.ndarray) -> np.ndarray:
        bounds = self.env_cfg["workspace_bounds"]
        return np.array(
            [
                np.clip(pos[0], *bounds["x"]),
                np.clip(pos[1], *bounds["y"]),
                np.clip(pos[2], *bounds["z"]),
            ],
            dtype=float,
        )

    def _world_to_pixel(self, xy: np.ndarray, width: int, height: int) -> tuple[int, int]:
        bounds = self.env_cfg["workspace_bounds"]
        x = (xy[0] - bounds["x"][0]) / (bounds["x"][1] - bounds["x"][0])
        y = (xy[1] - bounds["y"][0]) / (bounds["y"][1] - bounds["y"][0])
        return int(np.clip(x, 0, 1) * (width - 1)), int((1 - np.clip(y, 0, 1)) * (height - 1))
