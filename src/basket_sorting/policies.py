from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from basket_sorting.features import bc_feature_vector
from basket_sorting.fsm import ScriptedPickPlaceFSM


@dataclass
class PolicyOutput:
    action: np.ndarray
    phase: str


class LinearBCPolicy:
    """Small NumPy BC policy trained on state + subtask/phase features."""

    def __init__(self, model_path: str | Path, config: dict | None = None) -> None:
        data = np.load(model_path)
        self.weights = data["weights"]
        self.feature_mean = data["feature_mean"]
        self.feature_std = data["feature_std"]
        self.phase_tracker = ScriptedPickPlaceFSM(config) if config is not None else None

    def reset(self) -> None:
        if self.phase_tracker is not None:
            self.phase_tracker.reset()

    def act(self, obs: dict) -> PolicyOutput:
        phase = "linear_bc"
        if self.phase_tracker is not None:
            tracked = self.phase_tracker.act(obs)
            phase = tracked.phase
        features = bc_feature_vector(np.asarray(obs["state_features"], dtype=np.float32), phase)
        x = (features - self.feature_mean) / self.feature_std
        x_aug = np.concatenate([x, np.ones(1, dtype=np.float32)])
        action = x_aug @ self.weights
        action[:3] = np.clip(action[:3], -0.035, 0.035)
        action[3] = 1.0 if action[3] >= 0.0 else -1.0
        return PolicyOutput(action=action.astype(float), phase=phase)


def train_linear_bc(x: np.ndarray, y: np.ndarray, ridge_lambda: float) -> dict[str, np.ndarray]:
    feature_mean = x.mean(axis=0)
    feature_std = x.std(axis=0) + 1e-6
    x_norm = (x - feature_mean) / feature_std
    x_aug = np.concatenate([x_norm, np.ones((x_norm.shape[0], 1), dtype=x_norm.dtype)], axis=1)
    reg = ridge_lambda * np.eye(x_aug.shape[1], dtype=np.float64)
    weights = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)
    preds = x_aug @ weights
    mse = np.mean((preds - y) ** 2)
    return {
        "weights": weights.astype(np.float32),
        "feature_mean": feature_mean.astype(np.float32),
        "feature_std": feature_std.astype(np.float32),
        "train_mse": np.asarray(mse, dtype=np.float32),
    }


class TorchClipBCPolicy:
    """Placeholder for the proposal's frozen CLIP + MLP policy."""

    def __init__(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch/CLIP are not installed. Use train_bc.py for the NumPy state-feature "
                "baseline, or install PyTorch and add the CLIP implementation here."
            ) from exc
