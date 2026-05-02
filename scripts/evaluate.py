from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from basket_sorting.config import load_config
from basket_sorting.env_factory import make_env
from basket_sorting.evaluation import save_gif, save_json, summarize
from basket_sorting.fsm import ScriptedPickPlaceFSM
from basket_sorting.policies import LinearBCPolicy
from basket_sorting.rollout import run_episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FSM or BC policy.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--policy", choices=["fsm", "linear_bc"], default="fsm")
    parser.add_argument("--model", default="models/state_linear_bc.npz")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=500)
    parser.add_argument("--out", default="runs/eval.json")
    parser.add_argument("--save-video", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    results = []
    frames = []
    for episode in range(args.episodes):
        env = make_env(config, seed=args.seed + episode)
        policy = ScriptedPickPlaceFSM(config) if args.policy == "fsm" else LinearBCPolicy(args.model, config)
        result = run_episode(
            env,
            policy,
            seed=args.seed + episode,
            frames=bool(args.save_video) and episode < 5,
        )
        results.append(
            {
                "episode": episode,
                "success": result.success,
                "steps": result.steps,
                "instruction": result.instruction,
            }
        )
        frames.extend(result.frames)
        print(f"episode={episode} success={result.success} steps={result.steps}")

    summary = summarize(results)
    save_json(args.out, summary)
    if args.save_video:
        save_gif(args.save_video, frames)
    print(f"success_rate={summary['success_rate']:.3f} avg_steps={summary['avg_steps']:.2f} out={args.out}")


if __name__ == "__main__":
    main()
