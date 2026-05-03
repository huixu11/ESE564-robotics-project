from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from basket_sorting.config import load_config
from basket_sorting.env_factory import make_env
from basket_sorting.evaluation import save_gif
from basket_sorting.fsm import ScriptedPickPlaceFSM
from basket_sorting.rollout import run_episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scripted basket-sorting demo.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--save-video", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    all_frames = []
    successes = 0
    for episode in range(args.episodes):
        env = make_env(config, seed=args.seed + episode)
        policy = ScriptedPickPlaceFSM(config)
        try:
            result = run_episode(
                env,
                policy,
                instruction=args.instruction,
                seed=args.seed + episode,
                frames=bool(args.save_video),
            )
        finally:
            env.close()
        successes += int(result.success)
        all_frames.extend(result.frames)
        print(
            f"episode={episode} success={result.success} steps={result.steps} "
            f"instruction={result.instruction!r}"
        )

    if args.save_video:
        save_gif(args.save_video, all_frames)
        print(f"saved_video={args.save_video}")
    print(f"success_rate={successes / max(1, args.episodes):.3f}")


if __name__ == "__main__":
    main()
