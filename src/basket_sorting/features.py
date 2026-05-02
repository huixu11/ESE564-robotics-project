from __future__ import annotations

import numpy as np

from basket_sorting.fsm import PHASES


def phase_one_hot(phase: str) -> np.ndarray:
    vec = np.zeros(len(PHASES), dtype=np.float32)
    if phase in PHASES:
        vec[PHASES.index(phase)] = 1.0
    return vec


def bc_feature_vector(state_features: np.ndarray, phase: str) -> np.ndarray:
    """Feature vector for the NumPy BC baseline.

    Includes state, phase one-hot, and state-by-phase interactions so each
    subtask can learn its own linear motion field.
    """

    state = np.asarray(state_features, dtype=np.float32)
    phase_vec = phase_one_hot(phase)
    interactions = np.concatenate([state * value for value in phase_vec], axis=0)
    return np.concatenate([state, phase_vec, interactions], axis=0)
