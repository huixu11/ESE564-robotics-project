from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from basket_sorting.config import load_config
from basket_sorting.data import DemoWriter
from basket_sorting.env_factory import make_env
from basket_sorting.fsm import ScriptedPickPlaceFSM
from basket_sorting.rollout import run_episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect FSM demonstrations.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--out", default="data/demos")
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    writer = DemoWriter(
        args.out,
        save_images=bool(config["data"]["save_images"]) and not args.no_images,
        image_format=str(config["data"]["image_format"]),
    )
    saved = 0
    for episode in range(args.episodes):
        env = make_env(config, seed=args.seed + episode)
        policy = ScriptedPickPlaceFSM(config)
        try:
            result = run_episode(env, policy, seed=args.seed + episode, record=True)
        finally:
            env.close()
        if result.success or not config["data"]["keep_success_only"]:
            writer.write_episode(saved, result.instruction, result.records, result.success)
            saved += 1
        print(f"episode={episode} success={result.success} steps={result.steps} saved={saved}")
    print(f"saved_episodes={saved} out={args.out}")


if __name__ == "__main__":
    main()
