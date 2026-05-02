from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from basket_sorting.controllers import DifferentialIKController
from basket_sorting.tasks import TaskSpec, make_instruction, parse_instruction


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


@dataclass
class _ResolvedNames:
    arm_joint_ids: list[int]
    arm_qpos_addr: np.ndarray
    arm_dof_addr: np.ndarray
    arm_actuator_ids: list[int]
    gripper_actuator_ids: list[int]
    object_body_ids: dict[str, int]
    object_free_joint_ids: dict[str, int]
    object_free_qpos_addr: dict[str, int]
    basket_body_ids: dict[str, int]
    ee_site_id: int
    camera_id: int | None


class MujocoPandaKinematics:
    """MuJoCo FK/Jacobian backend for the shared differential IK controller."""

    def __init__(self, mujoco_module: Any, model: Any, data: Any, names: _ResolvedNames) -> None:
        self.mujoco = mujoco_module
        self.model = model
        self.data = data
        self.names = names

    def forward(self, qpos: np.ndarray) -> np.ndarray:
        self.data.qpos[self.names.arm_qpos_addr] = np.asarray(qpos, dtype=float)
        self.mujoco.mj_forward(self.model, self.data)
        return np.asarray(self.data.site_xpos[self.names.ee_site_id], dtype=float).copy()

    def jacobian(self, qpos: np.ndarray) -> np.ndarray:
        self.data.qpos[self.names.arm_qpos_addr] = np.asarray(qpos, dtype=float)
        self.mujoco.mj_forward(self.model, self.data)
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        self.mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.names.ee_site_id)
        return jacp[:, self.names.arm_dof_addr]


class MujocoBasketSortingEnv:
    """MuJoCo Panda basket-sorting adapter using the fallback env API."""

    def __init__(self, config: dict[str, Any], seed: int | None = None) -> None:
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError(
                "MuJoCo is not installed. Install it with `python -m pip install mujoco`, "
                "then run with `--config configs/mujoco_template.yaml` after adding assets."
            ) from exc

        self.mujoco = mujoco
        self.config = config
        self.env_cfg = config["env"]
        self.mj_cfg = self.env_cfg.get("mujoco", {})
        self.rng = np.random.default_rng(config.get("seed", 0) if seed is None else seed)
        self.model_xml = _resolve_project_path(self.mj_cfg.get("model_xml", "assets/mujoco/scene.xml"))
        if not self.model_xml.exists():
            raise FileNotFoundError(
                f"MuJoCo model XML not found: {self.model_xml}. Copy the class scene there "
                "or update env.mujoco.model_xml in your config."
            )

        self.model = mujoco.MjModel.from_xml_path(str(self.model_xml))
        self.data = mujoco.MjData(self.model)
        self.names = self._resolve_names()
        self.kinematics = MujocoPandaKinematics(mujoco, self.model, self.data, self.names)
        self.controller = DifferentialIKController(
            kinematics=self.kinematics,
            damping=config["controller"]["damping"],
            max_ee_step=config["controller"]["max_ee_step"],
            max_joint_step=config["controller"]["max_joint_step"],
            max_joint_jump=config["controller"]["max_joint_jump"],
            joint_limits=self._arm_joint_limits(),
        )
        self.renderer = self._make_renderer()
        self.task: TaskSpec | None = None
        self.steps = 0
        self.gripper_open = True
        self.last_ik_success = True
        self.attached_object: str | None = None
        self.phase = "reset"

    def reset(self, instruction: str | None = None, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.mujoco.mj_resetData(self.model, self.data)

        object_names = list(self.env_cfg["object_names"])
        target_object = str(self.rng.choice(object_names))
        target_basket = str(self.rng.choice(list(self.env_cfg["basket_poses"].keys())))
        speed_name = str(self.rng.choice(["careful", "normal", "fast"], p=[0.2, 0.6, 0.2]))
        instruction = instruction or make_instruction(target_object, target_basket, speed_name)
        self.task = parse_instruction(instruction)

        initial_qpos = np.asarray(self.mj_cfg.get("initial_qpos", [0.0] * len(self.names.arm_qpos_addr)), dtype=float)
        self._set_arm_qpos(initial_qpos)
        self._set_gripper(open_gripper=True)
        self._randomize_scene()
        self.mujoco.mj_forward(self.model, self.data)

        self.steps = 0
        self.gripper_open = True
        self.attached_object = None
        self.last_ik_success = True
        self.phase = "reset"
        return self._observation()

    def step(self, action: np.ndarray | list[float]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        action_arr = np.asarray(action, dtype=float)
        if action_arr.shape != (4,):
            raise ValueError(f"action must have shape (4,), got {action_arr.shape}")

        current_qpos = self._arm_qpos()
        ik = self.controller.solve(current_qpos, action_arr[:3])
        self.last_ik_success = ik.success
        target_qpos = ik.qpos if ik.success else current_qpos
        self._apply_arm_target(target_qpos)
        self._set_gripper(open_gripper=bool(action_arr[3] >= 0.0))

        if self.mj_cfg.get("direct_joint_position", False):
            self._update_attachment()
            self.mujoco.mj_forward(self.model, self.data)
        else:
            for _ in range(int(self.mj_cfg.get("control_substeps", 8))):
                self._update_attachment()
                self.mujoco.mj_step(self.model, self.data)
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
        if self.renderer is not None:
            try:
                camera = self.names.camera_id if self.names.camera_id is not None else None
                self.renderer.update_scene(self.data, camera=camera)
                return np.asarray(self.renderer.render(), dtype=np.uint8)
            except Exception:
                pass
        return self._render_top_down_fallback()

    def is_success(self) -> bool:
        if self.task is None:
            return False
        obj_pos = self._object_pos(self.task.target_object)
        basket = np.asarray(self.env_cfg["basket_poses"][self.task.target_basket], dtype=float)
        half = float(self.env_cfg["basket_half_extent"])
        inside_xy = np.all(np.abs(obj_pos[:2] - basket[:2]) <= half)
        released_or_low = obj_pos[2] <= max(0.08, float(self.env_cfg["place_z"]) + 0.03)
        return bool(inside_xy and released_or_low)

    def get_state_features(self) -> np.ndarray:
        if self.task is None:
            raise RuntimeError("Call reset before requesting features.")
        ee = self._ee_pos()
        target = self._object_pos(self.task.target_object)
        basket = np.asarray(self.env_cfg["basket_poses"][self.task.target_basket], dtype=float)
        attached = 1.0 if self.attached_object == self.task.target_object else 0.0
        return np.concatenate(
            [
                ee,
                target,
                basket,
                target - ee,
                basket - ee,
                np.array(
                    [
                        float(self.env_cfg["safe_z"]) - ee[2],
                        float(self.env_cfg["grasp_z"]) - ee[2],
                        float(self.env_cfg["place_z"]) - ee[2],
                        1.0 if self.gripper_open else 0.0,
                        self.task.speed,
                        attached,
                    ],
                    dtype=float,
                ),
            ]
        )

    def _observation(self) -> dict[str, Any]:
        if self.task is None:
            raise RuntimeError("Environment has not been reset.")
        objects = {name: self._object_pos(name) for name in self.env_cfg["object_names"]}
        return {
            "rgb": self.render_rgb(),
            "qpos": self._arm_qpos(),
            "ee_pos": self._ee_pos(),
            "gripper_open": self.gripper_open,
            "task": self.task,
            "objects": objects,
            "attached_object": self.attached_object,
            "state_features": self.get_state_features(),
            "phase": self.phase,
        }

    def _resolve_names(self) -> _ResolvedNames:
        arm_joint_ids = [self._name_id(self.mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.mj_cfg["arm_joint_names"]]
        arm_qpos_addr = np.asarray([self.model.jnt_qposadr[joint_id] for joint_id in arm_joint_ids], dtype=int)
        arm_dof_addr = np.asarray([self.model.jnt_dofadr[joint_id] for joint_id in arm_joint_ids], dtype=int)
        arm_actuator_ids = [
            self._name_id(self.mujoco.mjtObj.mjOBJ_ACTUATOR, name, required=False)
            for name in self.mj_cfg.get("arm_actuator_names", [])
        ]
        arm_actuator_ids = [idx for idx in arm_actuator_ids if idx >= 0]
        gripper_actuator_ids = [
            self._name_id(self.mujoco.mjtObj.mjOBJ_ACTUATOR, name, required=False)
            for name in self.mj_cfg.get("gripper_actuator_names", [])
        ]
        gripper_actuator_ids = [idx for idx in gripper_actuator_ids if idx >= 0]
        object_body_ids = {
            name: self._name_id(self.mujoco.mjtObj.mjOBJ_BODY, body_name)
            for name, body_name in self.mj_cfg.get("object_body_names", {}).items()
        }
        object_free_joint_ids: dict[str, int] = {}
        object_free_qpos_addr: dict[str, int] = {}
        for name, joint_name in self.mj_cfg.get("object_free_joint_names", {}).items():
            joint_id = self._name_id(self.mujoco.mjtObj.mjOBJ_JOINT, joint_name, required=False)
            if joint_id >= 0:
                object_free_joint_ids[name] = joint_id
                object_free_qpos_addr[name] = int(self.model.jnt_qposadr[joint_id])
        basket_body_ids = {
            name: self._name_id(self.mujoco.mjtObj.mjOBJ_BODY, body_name)
            for name, body_name in self.mj_cfg.get("basket_body_names", {}).items()
        }
        ee_site_id = self._name_id(self.mujoco.mjtObj.mjOBJ_SITE, self.mj_cfg["ee_site"])
        camera_name = self.mj_cfg.get("camera_name")
        camera_id = (
            self._name_id(self.mujoco.mjtObj.mjOBJ_CAMERA, camera_name, required=False)
            if camera_name
            else None
        )
        if camera_id is not None and camera_id < 0:
            camera_id = None
        return _ResolvedNames(
            arm_joint_ids=arm_joint_ids,
            arm_qpos_addr=arm_qpos_addr,
            arm_dof_addr=arm_dof_addr,
            arm_actuator_ids=arm_actuator_ids,
            gripper_actuator_ids=gripper_actuator_ids,
            object_body_ids=object_body_ids,
            object_free_joint_ids=object_free_joint_ids,
            object_free_qpos_addr=object_free_qpos_addr,
            basket_body_ids=basket_body_ids,
            ee_site_id=ee_site_id,
            camera_id=camera_id,
        )

    def _name_id(self, obj_type: Any, name: str, required: bool = True) -> int:
        obj_id = int(self.mujoco.mj_name2id(self.model, obj_type, name))
        if required and obj_id < 0:
            raise KeyError(f"MuJoCo object named {name!r} was not found in {self.model_xml}")
        return obj_id

    def _arm_joint_limits(self) -> list[list[float]]:
        limits: list[list[float]] = []
        fallback = self.config["controller"]["joint_limits"]
        for idx, joint_id in enumerate(self.names.arm_joint_ids):
            if bool(self.model.jnt_limited[joint_id]):
                limits.append([float(self.model.jnt_range[joint_id, 0]), float(self.model.jnt_range[joint_id, 1])])
            else:
                limits.append([float(fallback[idx][0]), float(fallback[idx][1])])
        return limits

    def _randomize_scene(self) -> None:
        positions: dict[str, np.ndarray] = {}
        for name in self.env_cfg["object_names"]:
            pos = self._sample_object_pos()
            while any(np.linalg.norm(pos[:2] - other[:2]) < 0.15 for other in positions.values()):
                pos = self._sample_object_pos()
            positions[name] = pos
            self._set_object_pos(name, pos)

        for name, pose in self.env_cfg["basket_poses"].items():
            self._set_basket_pos(name, np.asarray(pose, dtype=float))

    def _sample_object_pos(self) -> np.ndarray:
        bounds = self.env_cfg["table_bounds"]
        return np.array([self.rng.uniform(*bounds["x"]), self.rng.uniform(-0.22, 0.12), 0.04], dtype=float)

    def _arm_qpos(self) -> np.ndarray:
        return np.asarray(self.data.qpos[self.names.arm_qpos_addr], dtype=float).copy()

    def _set_arm_qpos(self, qpos: np.ndarray) -> None:
        qpos = np.asarray(qpos, dtype=float)
        if qpos.shape != self.names.arm_qpos_addr.shape:
            raise ValueError(f"Arm qpos must have shape {self.names.arm_qpos_addr.shape}, got {qpos.shape}")
        self.data.qpos[self.names.arm_qpos_addr] = qpos

    def _apply_arm_target(self, qpos: np.ndarray) -> None:
        if self.mj_cfg.get("direct_joint_position", False) or not self.names.arm_actuator_ids:
            self._set_arm_qpos(qpos)
            self.data.qvel[self.names.arm_dof_addr] = 0.0
            self.mujoco.mj_forward(self.model, self.data)
            return
        for actuator_id, value in zip(self.names.arm_actuator_ids, qpos):
            self.data.ctrl[actuator_id] = value

    def _set_gripper(self, open_gripper: bool) -> None:
        self.gripper_open = open_gripper
        ctrl_value = (
            float(self.mj_cfg.get("open_gripper_ctrl", 0.04))
            if open_gripper
            else float(self.mj_cfg.get("closed_gripper_ctrl", 0.0))
        )
        for actuator_id in self.names.gripper_actuator_ids:
            self.data.ctrl[actuator_id] = ctrl_value

    def _update_attachment(self) -> None:
        if self.task is None:
            return
        ee = self._ee_pos()
        if self.gripper_open:
            if self.attached_object is not None:
                self._set_object_pos(self.attached_object, np.array([ee[0], ee[1], 0.04], dtype=float))
                self.attached_object = None
            return

        if self.attached_object is None:
            target_pos = self._object_pos(self.task.target_object)
            xy_dist = np.linalg.norm(target_pos[:2] - ee[:2])
            z_dist = abs(float(ee[2]) - float(self.env_cfg["grasp_z"]))
            if xy_dist <= float(self.env_cfg["grasp_radius"]) and z_dist <= 0.06:
                self.attached_object = self.task.target_object

        if self.attached_object is not None:
            held_pos = np.array([ee[0], ee[1], max(0.04, ee[2] - 0.045)], dtype=float)
            self._set_object_pos(self.attached_object, held_pos)

    def _ee_pos(self) -> np.ndarray:
        self.mujoco.mj_forward(self.model, self.data)
        return np.asarray(self.data.site_xpos[self.names.ee_site_id], dtype=float).copy()

    def _object_pos(self, name: str) -> np.ndarray:
        body_id = self.names.object_body_ids[name]
        return np.asarray(self.data.xpos[body_id], dtype=float).copy()

    def _set_object_pos(self, name: str, pos: np.ndarray) -> None:
        if name in self.names.object_free_qpos_addr:
            addr = self.names.object_free_qpos_addr[name]
            joint_id = self.names.object_free_joint_ids[name]
            dof_addr = int(self.model.jnt_dofadr[joint_id])
            self.data.qpos[addr : addr + 7] = np.array([pos[0], pos[1], pos[2], 1.0, 0.0, 0.0, 0.0], dtype=float)
            self.data.qvel[dof_addr : dof_addr + 6] = 0.0
        else:
            body_id = self.names.object_body_ids[name]
            self.model.body_pos[body_id] = pos

    def _set_basket_pos(self, name: str, pos: np.ndarray) -> None:
        body_id = self.names.basket_body_ids.get(name)
        if body_id is not None:
            self.model.body_pos[body_id] = pos

    def _make_renderer(self) -> Any | None:
        try:
            return self.mujoco.Renderer(
                self.model,
                height=int(self.env_cfg["camera"]["height"]),
                width=int(self.env_cfg["camera"]["width"]),
            )
        except Exception:
            return None

    def _render_top_down_fallback(self) -> np.ndarray:
        width = int(self.env_cfg["camera"]["width"])
        height = int(self.env_cfg["camera"]["height"])
        image = Image.new("RGB", (width, height), (236, 232, 220))
        draw = ImageDraw.Draw(image)
        for name, center in self.env_cfg["basket_poses"].items():
            xy = np.asarray(center[:2], dtype=float)
            half = float(self.env_cfg["basket_half_extent"])
            p0 = self._world_to_pixel(xy - half, width, height)
            p1 = self._world_to_pixel(xy + half, width, height)
            box = [(min(p0[0], p1[0]), min(p0[1], p1[1])), (max(p0[0], p1[0]), max(p0[1], p1[1]))]
            color = (64, 130, 200) if name == "left" else (80, 165, 110)
            draw.rectangle(box, outline=color, width=3)
        colors = {"cracker_box": (205, 80, 60), "mustard_bottle": (225, 190, 55)}
        for name in self.env_cfg["object_names"]:
            px, py = self._world_to_pixel(self._object_pos(name)[:2], width, height)
            draw.ellipse([(px - 7, py - 7), (px + 7, py + 7)], fill=colors.get(name, (130, 130, 130)))
        ex, ey = self._world_to_pixel(self._ee_pos()[:2], width, height)
        draw.line([(ex - 8, ey), (ex + 8, ey)], fill=(30, 30, 30), width=2)
        draw.line([(ex, ey - 8), (ex, ey + 8)], fill=(30, 30, 30), width=2)
        return np.asarray(image)

    def _world_to_pixel(self, xy: np.ndarray, width: int, height: int) -> tuple[int, int]:
        bounds = self.env_cfg["workspace_bounds"]
        x = (xy[0] - bounds["x"][0]) / (bounds["x"][1] - bounds["x"][0])
        y = (xy[1] - bounds["y"][0]) / (bounds["y"][1] - bounds["y"][0])
        return int(np.clip(x, 0, 1) * (width - 1)), int((1 - np.clip(y, 0, 1)) * (height - 1))
