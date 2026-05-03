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
from basket_sorting.perception import estimate_color_positions
from basket_sorting.push_fsm import ScriptedPushFSM
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

    def test_color_perception_estimates_centroids(self) -> None:
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        image[8:13, 10:15] = np.array([200, 40, 40], dtype=np.uint8)
        image[20:25, 28:33] = np.array([220, 170, 40], dtype=np.uint8)
        cfg = {
            "min_pixels": 5,
            "image_to_world_affine": [
                [0.01, 0.0, 0.0],
                [0.0, 0.01, 0.0],
            ],
        }
        estimates = estimate_color_positions(image, cfg)
        self.assertIn("cracker_box", estimates)
        self.assertIn("mustard_bottle", estimates)
        self.assertTrue(np.allclose(estimates["cracker_box"][:2], [0.12, 0.10], atol=0.02))
        self.assertTrue(np.allclose(estimates["mustard_bottle"][:2], [0.30, 0.22], atol=0.02))

    def test_push_fsm_uses_observed_object_and_basket(self) -> None:
        config = load_config()
        policy = ScriptedPushFSM(config)
        obs = {
            "task": parse_instruction("place the cracker box in the left basket"),
            "ee_pos": np.array([0.0, 0.0, 0.28], dtype=float),
            "objects": {"cracker_box": np.array([0.0, 0.0, 0.04], dtype=float)},
            "baskets": {"left": np.array([0.0, 0.25, 0.0], dtype=float)},
        }
        output = policy.act(obs)
        self.assertEqual(output.phase, "approach_push_start")
        self.assertEqual(output.action.shape, (4,))

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
