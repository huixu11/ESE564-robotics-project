from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class KinematicsBackend(Protocol):
    def forward(self, qpos: np.ndarray) -> np.ndarray:
        ...

    def jacobian(self, qpos: np.ndarray) -> np.ndarray:
        ...


class ToyPandaKinematics:
    """Small deterministic backend used until the class Panda wrapper is added.

    The first three coordinates encode end-effector xyz directly. The remaining
    coordinates are redundant null-space joints. This keeps the controller API
    realistic while making local tests runnable without MuJoCo.
    """

    def forward(self, qpos: np.ndarray) -> np.ndarray:
        return np.asarray(qpos[:3], dtype=float)

    def jacobian(self, qpos: np.ndarray) -> np.ndarray:
        jac = np.zeros((3, len(qpos)), dtype=float)
        jac[0, 0] = 1.0
        jac[1, 1] = 1.0
        jac[2, 2] = 1.0
        return jac


@dataclass
class IKResult:
    qpos: np.ndarray
    ee_delta: np.ndarray
    success: bool
    reason: str = "ok"


class DifferentialIKController:
    """Damped least-squares differential IK with per-step continuity guards."""

    def __init__(
        self,
        kinematics: KinematicsBackend,
        damping: float,
        max_ee_step: float,
        max_joint_step: float,
        max_joint_jump: float,
        joint_limits: list[list[float]],
    ) -> None:
        self.kinematics = kinematics
        self.damping = float(damping)
        self.max_ee_step = float(max_ee_step)
        self.max_joint_step = float(max_joint_step)
        self.max_joint_jump = float(max_joint_jump)
        self.joint_limits = np.asarray(joint_limits, dtype=float)

    def solve(self, qpos: np.ndarray, ee_delta: np.ndarray) -> IKResult:
        qpos = np.asarray(qpos, dtype=float)
        ee_delta = np.asarray(ee_delta, dtype=float)
        if ee_delta.shape != (3,):
            raise ValueError(f"ee_delta must have shape (3,), got {ee_delta.shape}")

        requested_norm = float(np.linalg.norm(ee_delta))
        if requested_norm > self.max_ee_step:
            ee_delta = ee_delta / requested_norm * self.max_ee_step

        jac = self.kinematics.jacobian(qpos)
        jj_t = jac @ jac.T
        damping_matrix = (self.damping**2) * np.eye(jj_t.shape[0])
        dq = jac.T @ np.linalg.solve(jj_t + damping_matrix, ee_delta)

        dq = np.clip(dq, -self.max_joint_step, self.max_joint_step)
        if np.max(np.abs(dq)) > self.max_joint_jump:
            return IKResult(qpos=qpos.copy(), ee_delta=np.zeros(3), success=False, reason="joint_jump")

        next_qpos = qpos + dq
        if self.joint_limits.shape == (len(qpos), 2):
            next_qpos = np.clip(next_qpos, self.joint_limits[:, 0], self.joint_limits[:, 1])

        applied_delta = self.kinematics.forward(next_qpos) - self.kinematics.forward(qpos)
        return IKResult(qpos=next_qpos, ee_delta=applied_delta, success=True)
