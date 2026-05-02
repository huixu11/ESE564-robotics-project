from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from basket_sorting.config import load_config
from basket_sorting.controllers import DifferentialIKController, ToyPandaKinematics
from basket_sorting.env_factory import make_env
from basket_sorting.fsm import ScriptedPickPlaceFSM
from basket_sorting.rollout import run_episode
from basket_sorting.tasks import parse_instruction


class BaselineTests(unittest.TestCase):
    def test_parse_instruction(self) -> None:
        task = parse_instruction("fast place the mustard bottle in the right basket")
        self.assertEqual(task.target_object, "mustard_bottle")
        self.assertEqual(task.target_basket, "right")
        self.assertEqual(task.speed_name, "fast")

    def test_differential_ik_clamps_joint_steps(self) -> None:
        config = load_config()
        controller = DifferentialIKController(
            ToyPandaKinematics(),
            damping=config["controller"]["damping"],
            max_ee_step=config["controller"]["max_ee_step"],
            max_joint_step=config["controller"]["max_joint_step"],
            max_joint_jump=config["controller"]["max_joint_jump"],
            joint_limits=config["controller"]["joint_limits"],
        )
        qpos = np.array([0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0])
        result = controller.solve(qpos, np.array([1.0, 1.0, 1.0]))
        self.assertTrue(result.success)
        self.assertLessEqual(np.max(np.abs(result.qpos - qpos)), config["controller"]["max_joint_step"] + 1e-9)

    def test_scripted_fsm_completes_fixed_seed(self) -> None:
        config = load_config()
        env = make_env(config, seed=11)
        policy = ScriptedPickPlaceFSM(config)
        result = run_episode(env, policy, seed=11)
        self.assertTrue(result.success)
        self.assertLess(result.steps, config["env"]["max_steps"])

    def test_mujoco_config_fails_clearly_when_dependency_missing(self) -> None:
        config = load_config()
        config["env"]["name"] = "mujoco"
        try:
            import mujoco  # noqa: F401
        except ImportError:
            with self.assertRaisesRegex(RuntimeError, "MuJoCo is not installed"):
                make_env(config)


if __name__ == "__main__":
    unittest.main()
