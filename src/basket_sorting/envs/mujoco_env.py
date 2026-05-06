from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from basket_sorting.controllers import DifferentialIKController
from basket_sorting.perception import estimate_color_positions
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
    gripper_qpos_addr: np.ndarray
    object_body_ids: dict[str, int]
    object_free_joint_ids: dict[str, int]
    object_free_qpos_addr: dict[str, int]
    basket_body_ids: dict[str, int]
    ee_site_id: int | None
    ee_body_id: int | None
    camera_id: int | None
    perception_camera_id: int | None
    contact_pusher_body_id: int | None
    contact_pusher_mocap_id: int | None
    grasp_tool_body_id: int | None
    grasp_tool_mocap_id: int | None
    grasp_tool_qpos_addr: np.ndarray
    base_qpos_addr: int | None
    base_dof_addr: int | None


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
        if self.names.ee_site_id is not None:
            return np.asarray(self.data.site_xpos[self.names.ee_site_id], dtype=float).copy()
        if self.names.ee_body_id is not None:
            return np.asarray(self.data.xpos[self.names.ee_body_id], dtype=float).copy()
        raise RuntimeError("No end-effector site or body was configured.")

    def jacobian(self, qpos: np.ndarray) -> np.ndarray:
        self.data.qpos[self.names.arm_qpos_addr] = np.asarray(qpos, dtype=float)
        self.mujoco.mj_forward(self.model, self.data)
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        if self.names.ee_site_id is not None:
            self.mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.names.ee_site_id)
        elif self.names.ee_body_id is not None:
            self.mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.names.ee_body_id)
        else:
            raise RuntimeError("No end-effector site or body was configured.")
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
        self._apply_option_overrides()
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
        self.last_perceived_objects: dict[str, np.ndarray] = {}
        self.last_perceived_baskets: dict[str, np.ndarray] = {}
        self.last_contact_pusher_pos: np.ndarray | None = None
        self.last_grasp_tool_pos: np.ndarray | None = None
        self.base_qpos0: np.ndarray | None = None
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
        self.last_perceived_objects = {}
        self.last_perceived_baskets = {}
        self.last_contact_pusher_pos = None
        self.last_grasp_tool_pos = None
        self.base_qpos0 = self._base_qpos().copy()
        self.last_ik_success = True
        self.phase = "reset"
        self._sync_contact_pusher(initial=True)
        self._sync_grasp_tool(initial=True)
        self._settle_scene(int(self.mj_cfg.get("reset_settle_steps", 0)), initial_qpos)
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
            if self._push_proxy_enabled() and self.phase == "push":
                for _ in range(int(self.mj_cfg.get("control_substeps", 8))):
                    self._apply_push_proxy(action_arr)
                    self.mujoco.mj_step(self.model, self.data)
                self.data.xfrc_applied[:] = 0.0
                self.data.qfrc_applied[:] = 0.0
            elif self._contact_pusher_enabled():
                start_pos = self._contact_pusher_pos()
                target_pos = self._ee_pos()
                for _ in range(int(self.mj_cfg.get("control_substeps", 8))):
                    alpha = float(_ + 1) / float(int(self.mj_cfg.get("control_substeps", 8)))
                    self._lock_direct_robot_state(target_qpos)
                    self._set_contact_pusher_pos(start_pos * (1.0 - alpha) + target_pos * alpha)
                    self.mujoco.mj_step(self.model, self.data)
                    self._lock_direct_robot_state(target_qpos)
                self.last_contact_pusher_pos = target_pos.copy()
                self.mujoco.mj_forward(self.model, self.data)
            elif self._grasp_tool_enabled():
                start_pos = self._grasp_tool_pos()
                target_pos = self._ee_pos()
                substeps = int(self.mj_cfg.get("control_substeps", 8))
                for step_idx in range(substeps):
                    alpha = float(step_idx + 1) / float(substeps)
                    self._lock_direct_robot_state(target_qpos)
                    self._set_grasp_tool_opening()
                    self._set_grasp_tool_pos(start_pos * (1.0 - alpha) + target_pos * alpha)
                    self.mujoco.mj_step(self.model, self.data)
                    self._lock_direct_robot_state(target_qpos)
                    self._set_grasp_tool_opening()
                self.last_grasp_tool_pos = target_pos.copy()
                self.mujoco.mj_forward(self.model, self.data)
            else:
                self._update_attachment()
                self.mujoco.mj_forward(self.model, self.data)
        else:
            for _ in range(int(self.mj_cfg.get("control_substeps", 8))):
                if self._push_proxy_enabled():
                    self._apply_push_proxy(action_arr)
                else:
                    self._update_attachment()
                self.mujoco.mj_step(self.model, self.data)
            if self._push_proxy_enabled():
                self.data.xfrc_applied[:] = 0.0
                self.data.qfrc_applied[:] = 0.0
            else:
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
        if self.env_cfg.get("contact_debug", {}).get("enabled", False):
            info.update(self._contact_debug_info())
        return self._observation(), reward, done, info

    def render_rgb(self) -> np.ndarray:
        return self._render_camera(self.names.camera_id)

    def _render_camera(self, camera_id: int | None) -> np.ndarray:
        if self.renderer is not None:
            try:
                self.renderer.update_scene(self.data, camera=camera_id)
                return np.asarray(self.renderer.render(), dtype=np.uint8)
            except Exception:
                pass
        return self._render_top_down_fallback()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def is_success(self) -> bool:
        if self.task is None:
            return False
        obj_pos = self._object_pos(self.task.target_object)
        basket = np.asarray(self.env_cfg["basket_poses"][self.task.target_basket], dtype=float)
        half = float(self.env_cfg["basket_half_extent"])
        inside_xy = np.all(np.abs(obj_pos[:2] - basket[:2]) <= half)
        released_or_low = obj_pos[2] <= max(0.08, float(self.env_cfg["place_z"]) + 0.03)
        return bool(inside_xy and released_or_low)

    def get_state_features(
        self,
        objects: dict[str, np.ndarray] | None = None,
        baskets: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        if self.task is None:
            raise RuntimeError("Call reset before requesting features.")
        ee = self._ee_pos()
        objects = objects or {name: self._object_pos(name) for name in self.env_cfg["object_names"]}
        baskets = baskets or {name: np.asarray(pose, dtype=float) for name, pose in self.env_cfg["basket_poses"].items()}
        target = np.asarray(objects[self.task.target_object], dtype=float)
        basket = np.asarray(baskets[self.task.target_basket], dtype=float)
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
        sim_objects = {name: self._object_pos(name) for name in self.env_cfg["object_names"]}
        sim_baskets = {name: np.asarray(pose, dtype=float) for name, pose in self.env_cfg["basket_poses"].items()}
        rgb = self.render_rgb()
        objects, baskets = self._policy_scene_estimate(rgb, sim_objects, sim_baskets)
        return {
            "rgb": rgb,
            "qpos": self._arm_qpos(),
            "ee_pos": self._ee_pos(),
            "gripper_open": self.gripper_open,
            "task": self.task,
            "objects": objects,
            "baskets": baskets,
            "sim_objects": sim_objects,
            "sim_baskets": sim_baskets,
            "attached_object": self.attached_object,
            "state_features": self.get_state_features(objects, baskets),
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
        gripper_joint_ids = [
            self._name_id(self.mujoco.mjtObj.mjOBJ_JOINT, name, required=False)
            for name in self.mj_cfg.get("gripper_joint_names", [])
        ]
        gripper_joint_ids = [idx for idx in gripper_joint_ids if idx >= 0]
        gripper_qpos_addr = np.asarray([self.model.jnt_qposadr[joint_id] for joint_id in gripper_joint_ids], dtype=int)
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
        ee_site_name = self.mj_cfg.get("ee_site")
        ee_body_name = self.mj_cfg.get("ee_body")
        ee_site_id = (
            self._name_id(self.mujoco.mjtObj.mjOBJ_SITE, ee_site_name, required=False)
            if ee_site_name
            else -1
        )
        ee_body_id = (
            self._name_id(self.mujoco.mjtObj.mjOBJ_BODY, ee_body_name, required=False)
            if ee_body_name
            else -1
        )
        if ee_site_id < 0 and ee_body_id < 0:
            raise KeyError(
                "Configured end-effector was not found. Set env.mujoco.ee_site or env.mujoco.ee_body "
                f"to a valid MuJoCo site/body name in {self.model_xml}."
            )
        camera_name = self.mj_cfg.get("camera_name")
        camera_id = (
            self._name_id(self.mujoco.mjtObj.mjOBJ_CAMERA, camera_name, required=False)
            if camera_name
            else None
        )
        if camera_id is not None and camera_id < 0:
            camera_id = None
        perception_camera_name = self.env_cfg.get("perception", {}).get("camera_name", camera_name)
        perception_camera_id = (
            self._name_id(self.mujoco.mjtObj.mjOBJ_CAMERA, perception_camera_name, required=False)
            if perception_camera_name
            else camera_id
        )
        if perception_camera_id is not None and perception_camera_id < 0:
            perception_camera_id = camera_id
        contact_pusher_name = self.mj_cfg.get("contact_pusher_body")
        contact_pusher_body_id = (
            self._name_id(self.mujoco.mjtObj.mjOBJ_BODY, contact_pusher_name, required=False)
            if contact_pusher_name
            else -1
        )
        contact_pusher_mocap_id = None
        if contact_pusher_body_id >= 0:
            mocap_id = int(self.model.body_mocapid[contact_pusher_body_id])
            contact_pusher_mocap_id = mocap_id if mocap_id >= 0 else None
        grasp_tool_name = self.mj_cfg.get("grasp_tool_body")
        grasp_tool_body_id = (
            self._name_id(self.mujoco.mjtObj.mjOBJ_BODY, grasp_tool_name, required=False)
            if grasp_tool_name
            else -1
        )
        grasp_tool_mocap_id = None
        if grasp_tool_body_id >= 0:
            mocap_id = int(self.model.body_mocapid[grasp_tool_body_id])
            grasp_tool_mocap_id = mocap_id if mocap_id >= 0 else None
        grasp_tool_joint_ids = [
            self._name_id(self.mujoco.mjtObj.mjOBJ_JOINT, name, required=False)
            for name in self.mj_cfg.get("grasp_tool_joint_names", [])
        ]
        grasp_tool_joint_ids = [idx for idx in grasp_tool_joint_ids if idx >= 0]
        grasp_tool_qpos_addr = np.asarray([self.model.jnt_qposadr[joint_id] for joint_id in grasp_tool_joint_ids], dtype=int)
        base_joint_name = self.mj_cfg.get("base_joint_name", "floating_base")
        base_joint_id = self._name_id(self.mujoco.mjtObj.mjOBJ_JOINT, base_joint_name, required=False)
        base_qpos_addr = int(self.model.jnt_qposadr[base_joint_id]) if base_joint_id >= 0 else None
        base_dof_addr = int(self.model.jnt_dofadr[base_joint_id]) if base_joint_id >= 0 else None
        return _ResolvedNames(
            arm_joint_ids=arm_joint_ids,
            arm_qpos_addr=arm_qpos_addr,
            arm_dof_addr=arm_dof_addr,
            arm_actuator_ids=arm_actuator_ids,
            gripper_actuator_ids=gripper_actuator_ids,
            gripper_qpos_addr=gripper_qpos_addr,
            object_body_ids=object_body_ids,
            object_free_joint_ids=object_free_joint_ids,
            object_free_qpos_addr=object_free_qpos_addr,
            basket_body_ids=basket_body_ids,
            ee_site_id=ee_site_id if ee_site_id >= 0 else None,
            ee_body_id=ee_body_id if ee_body_id >= 0 else None,
            camera_id=camera_id,
            perception_camera_id=perception_camera_id,
            contact_pusher_body_id=contact_pusher_body_id if contact_pusher_body_id >= 0 else None,
            contact_pusher_mocap_id=contact_pusher_mocap_id,
            grasp_tool_body_id=grasp_tool_body_id if grasp_tool_body_id >= 0 else None,
            grasp_tool_mocap_id=grasp_tool_mocap_id,
            grasp_tool_qpos_addr=grasp_tool_qpos_addr,
            base_qpos_addr=base_qpos_addr,
            base_dof_addr=base_dof_addr,
        )

    def _name_id(self, obj_type: Any, name: str, required: bool = True) -> int:
        obj_id = int(self.mujoco.mj_name2id(self.model, obj_type, name))
        if required and obj_id < 0:
            raise KeyError(f"MuJoCo object named {name!r} was not found in {self.model_xml}")
        return obj_id

    def _apply_option_overrides(self) -> None:
        for name, value in self.mj_cfg.get("option", {}).items():
            if not hasattr(self.model.opt, name):
                raise KeyError(f"MuJoCo option {name!r} is not available on this model.")
            setattr(self.model.opt, name, value)

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
        min_distance = float(self.env_cfg.get("object_min_distance", 0.15))
        for name in self.env_cfg["object_names"]:
            pos = self._sample_object_pos(name)
            while any(np.linalg.norm(pos[:2] - other[:2]) < min_distance for other in positions.values()):
                pos = self._sample_object_pos(name)
            positions[name] = pos
            self._set_object_pos(name, pos)

        for name, pose in self.env_cfg["basket_poses"].items():
            self._set_basket_pos(name, np.asarray(pose, dtype=float))

    def _sample_object_pos(self, name: str) -> np.ndarray:
        bounds = self.env_cfg["table_bounds"]
        spawn_z = float(self.env_cfg.get("object_spawn_z", {}).get(name, 0.04))
        return np.array([self.rng.uniform(*bounds["x"]), self.rng.uniform(*bounds["y"]), spawn_z], dtype=float)

    def _arm_qpos(self) -> np.ndarray:
        return np.asarray(self.data.qpos[self.names.arm_qpos_addr], dtype=float).copy()

    def _base_qpos(self) -> np.ndarray:
        if self.names.base_qpos_addr is None:
            return np.empty(0, dtype=float)
        addr = self.names.base_qpos_addr
        return np.asarray(self.data.qpos[addr : addr + 7], dtype=float).copy()

    def _set_arm_qpos(self, qpos: np.ndarray) -> None:
        qpos = np.asarray(qpos, dtype=float)
        if qpos.shape != self.names.arm_qpos_addr.shape:
            raise ValueError(f"Arm qpos must have shape {self.names.arm_qpos_addr.shape}, got {qpos.shape}")
        self.data.qpos[self.names.arm_qpos_addr] = qpos

    def _lock_direct_robot_state(self, arm_qpos: np.ndarray) -> None:
        self._set_arm_qpos(arm_qpos)
        self.data.qvel[self.names.arm_dof_addr] = 0.0
        if self.names.base_qpos_addr is not None and self.base_qpos0 is not None:
            qaddr = self.names.base_qpos_addr
            daddr = self.names.base_dof_addr
            self.data.qpos[qaddr : qaddr + 7] = self.base_qpos0
            if daddr is not None:
                self.data.qvel[daddr : daddr + 6] = 0.0
        if len(self.names.gripper_qpos_addr):
            ctrl_value = float(self.mj_cfg.get("open_gripper_ctrl", 0.04))
            if not self.gripper_open:
                ctrl_value = float(self.mj_cfg.get("closed_gripper_ctrl", 0.0))
            self.data.qpos[self.names.gripper_qpos_addr] = ctrl_value
        self._set_grasp_tool_opening()

    def _settle_scene(self, steps: int, arm_qpos: np.ndarray) -> None:
        for _ in range(max(0, int(steps))):
            if self.mj_cfg.get("direct_joint_position", False):
                self._lock_direct_robot_state(arm_qpos)
            self.mujoco.mj_step(self.model, self.data)
            if self.mj_cfg.get("direct_joint_position", False):
                self._lock_direct_robot_state(arm_qpos)
        self.mujoco.mj_forward(self.model, self.data)

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
        if self.mj_cfg.get("direct_joint_position", False) and len(self.names.gripper_qpos_addr):
            self.data.qpos[self.names.gripper_qpos_addr] = ctrl_value
        self._set_grasp_tool_opening()
        if self.mj_cfg.get("direct_joint_position", False) and (len(self.names.gripper_qpos_addr) or len(self.names.grasp_tool_qpos_addr)):
            self.mujoco.mj_forward(self.model, self.data)

    def _push_proxy_enabled(self) -> bool:
        return bool(self.env_cfg.get("physics_push", {}).get("enabled", False))

    def _contact_pusher_enabled(self) -> bool:
        return self.names.contact_pusher_mocap_id is not None

    def _grasp_tool_enabled(self) -> bool:
        return self.names.grasp_tool_mocap_id is not None

    def _contact_pusher_pos(self) -> np.ndarray:
        if self.names.contact_pusher_mocap_id is None:
            return self._ee_pos()
        if self.last_contact_pusher_pos is not None:
            return self.last_contact_pusher_pos.copy()
        return np.asarray(self.data.mocap_pos[self.names.contact_pusher_mocap_id], dtype=float).copy()

    def _set_contact_pusher_pos(self, pos: np.ndarray) -> None:
        if self.names.contact_pusher_mocap_id is None:
            return
        self.data.mocap_pos[self.names.contact_pusher_mocap_id] = np.asarray(pos, dtype=float)
        self.data.mocap_quat[self.names.contact_pusher_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    def _sync_contact_pusher(self, initial: bool = False) -> None:
        if self.names.contact_pusher_mocap_id is None:
            return
        target_pos = self._ee_pos()
        self._set_contact_pusher_pos(target_pos)
        if initial:
            self.last_contact_pusher_pos = target_pos.copy()

    def _grasp_tool_pos(self) -> np.ndarray:
        if self.names.grasp_tool_mocap_id is None:
            return self._ee_pos()
        if self.last_grasp_tool_pos is not None:
            return self.last_grasp_tool_pos.copy()
        return np.asarray(self.data.mocap_pos[self.names.grasp_tool_mocap_id], dtype=float).copy()

    def _set_grasp_tool_pos(self, pos: np.ndarray) -> None:
        if self.names.grasp_tool_mocap_id is None:
            return
        self.data.mocap_pos[self.names.grasp_tool_mocap_id] = np.asarray(pos, dtype=float)
        self.data.mocap_quat[self.names.grasp_tool_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    def _set_grasp_tool_opening(self) -> None:
        if not len(self.names.grasp_tool_qpos_addr):
            return
        key = "grasp_tool_open_qpos" if self.gripper_open else "grasp_tool_closed_qpos"
        values = np.asarray(self.mj_cfg.get(key, [0.0] * len(self.names.grasp_tool_qpos_addr)), dtype=float)
        if values.shape != self.names.grasp_tool_qpos_addr.shape:
            raise ValueError(
                f"env.mujoco.{key} must have shape {self.names.grasp_tool_qpos_addr.shape}, got {values.shape}"
            )
        self.data.qpos[self.names.grasp_tool_qpos_addr] = values

    def _sync_grasp_tool(self, initial: bool = False) -> None:
        if self.names.grasp_tool_mocap_id is None:
            return
        target_pos = self._ee_pos()
        self._set_grasp_tool_pos(target_pos)
        self._set_grasp_tool_opening()
        if initial:
            self.last_grasp_tool_pos = target_pos.copy()

    def _apply_push_proxy(self, action_arr: np.ndarray) -> None:
        self.data.xfrc_applied[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        if self.task is None or not self._push_proxy_enabled() or self.phase != "push":
            return
        target_pos = self._object_pos(self.task.target_object)
        basket = np.asarray(self.env_cfg["basket_poses"][self.task.target_basket], dtype=float)
        direction = basket[:2] - target_pos[:2]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            return
        direction = direction / norm

        ee = self._ee_pos()
        xy_dist = float(np.linalg.norm(target_pos[:2] - ee[:2]))
        push_cfg = self.env_cfg.get("physics_push", {})
        if xy_dist > float(push_cfg.get("radius", 0.13)):
            return

        force = float(push_cfg.get("force", 38.0))
        joint_id = self.names.object_free_joint_ids.get(self.task.target_object)
        if joint_id is not None:
            dof_addr = int(self.model.jnt_dofadr[joint_id])
            velocity = float(push_cfg.get("velocity", 0.0))
            if velocity > 0.0:
                self.data.qvel[dof_addr] = velocity * direction[0]
                self.data.qvel[dof_addr + 1] = velocity * direction[1]
                self.data.qvel[dof_addr + 2] = 0.0
                self.data.qvel[dof_addr + 3 : dof_addr + 6] = 0.0
            self.data.qfrc_applied[dof_addr] = force * direction[0]
            self.data.qfrc_applied[dof_addr + 1] = force * direction[1]
        else:
            body_id = self.names.object_body_ids[self.task.target_object]
            self.data.xfrc_applied[body_id, 0] = force * direction[0]
            self.data.xfrc_applied[body_id, 1] = force * direction[1]

    def _contact_debug_info(self) -> dict[str, Any]:
        pairs = []
        max_pairs = int(self.env_cfg.get("contact_debug", {}).get("max_pairs", 12))
        for idx in range(min(int(self.data.ncon), max_pairs)):
            contact = self.data.contact[idx]
            geom1 = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
            geom2 = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
            pairs.append([geom1 or str(int(contact.geom1)), geom2 or str(int(contact.geom2))])
        info: dict[str, Any] = {
            "num_contacts": int(self.data.ncon),
            "contact_pairs": pairs,
        }
        if self.task is not None:
            info["target_object_pos"] = self._object_pos(self.task.target_object).tolist()
        if self._contact_pusher_enabled():
            info["contact_pusher_pos"] = self._contact_pusher_pos().tolist()
        return info

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

    def _policy_scene_estimate(
        self,
        rgb: np.ndarray,
        sim_objects: dict[str, np.ndarray],
        sim_baskets: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        perception_cfg = self.env_cfg.get("perception", {})
        if not perception_cfg.get("enabled", False):
            return sim_objects, sim_baskets

        perception_rgb = rgb
        if self.names.perception_camera_id != self.names.camera_id:
            perception_rgb = self._render_camera(self.names.perception_camera_id)
        estimates = estimate_color_positions(perception_rgb, perception_cfg)

        objects: dict[str, np.ndarray] = {}
        for name in self.env_cfg["object_names"]:
            if name in estimates:
                self.last_perceived_objects[name] = estimates[name]
            if name in self.last_perceived_objects:
                objects[name] = self.last_perceived_objects[name].copy()

        baskets: dict[str, np.ndarray] = {}
        for name, pose in sim_baskets.items():
            if name in estimates:
                self.last_perceived_baskets[name] = estimates[name]
            # Basket locations are fixed goal regions in the scene map; object
            # positions are the variable state estimated from camera color.
            baskets[name] = np.asarray(pose, dtype=float).copy()

        if len(objects) != len(self.env_cfg["object_names"]):
            missing = set(self.env_cfg["object_names"]) - set(objects)
            for name in missing:
                objects[name] = sim_objects[name].copy()
        return objects, baskets

    def _ee_pos(self) -> np.ndarray:
        self.mujoco.mj_forward(self.model, self.data)
        if self.names.ee_site_id is not None:
            return np.asarray(self.data.site_xpos[self.names.ee_site_id], dtype=float).copy()
        if self.names.ee_body_id is not None:
            return np.asarray(self.data.xpos[self.names.ee_body_id], dtype=float).copy()
        raise RuntimeError("No end-effector site or body was configured.")

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
