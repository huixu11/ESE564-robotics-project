from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from basket_sorting.config import load_config
from basket_sorting.env_factory import make_env
from basket_sorting.evaluation import save_gif, save_json, summarize
from basket_sorting.fsm import ScriptedPickPlaceFSM
from basket_sorting.policies import LinearBCPolicy
from basket_sorting.push_fsm import ScriptedPushFSM
from basket_sorting.rollout import run_episode
from basket_sorting.tamp_grasp import ScriptedTAMPGraspPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FSM or BC policy.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--policy", choices=["fsm", "push_fsm", "tamp_grasp", "linear_bc"], default="fsm")
    parser.add_argument("--model", default="models/state_linear_bc.npz")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=500)
    parser.add_argument(
        "--instruction",
        default=None,
        help="Optional language command to execute in every episode, e.g. 'place the mustard bottle in the right basket'.",
    )
    parser.add_argument("--out", default="runs/eval.json")
    parser.add_argument("--save-video", default=None)
    parser.add_argument("--video-episodes", type=int, default=5)
    parser.add_argument("--video-stride", type=int, default=1)
    parser.add_argument("--max-video-frames", type=int, default=3000)
    args = parser.parse_args()

    config = load_config(args.config)
    results = []
    frames = []
    frame_captions = []
    for episode in range(args.episodes):
        env = make_env(config, seed=args.seed + episode)
        if args.policy == "fsm":
            policy = ScriptedPickPlaceFSM(config)
        elif args.policy == "push_fsm":
            policy = ScriptedPushFSM(config)
        elif args.policy == "tamp_grasp":
            policy = ScriptedTAMPGraspPolicy(config)
        else:
            policy = LinearBCPolicy(args.model, config)
        try:
            result = run_episode(
                env,
                policy,
                instruction=args.instruction,
                seed=args.seed + episode,
                frames=bool(args.save_video) and episode < args.video_episodes,
            )
        finally:
            env.close()
        results.append(
            {
                "episode": episode,
                "success": result.success,
                "steps": result.steps,
                "instruction": result.instruction,
            }
        )
        frames.extend(result.frames)
        frame_captions.extend([result.instruction] * len(result.frames))
        print(
            f"episode={episode} success={result.success} steps={result.steps} "
            f"instruction={result.instruction!r}"
        )

    summary = summarize(results)
    save_json(args.out, summary)
    if args.save_video:
        saved_frames = save_gif(
            args.save_video,
            frames,
            frame_stride=args.video_stride,
            max_frames=args.max_video_frames,
            captions=frame_captions,
        )
        print(f"saved_video={args.save_video} frames={saved_frames} raw_frames={len(frames)}")
    print(f"success_rate={summary['success_rate']:.3f} avg_steps={summary['avg_steps']:.2f} out={args.out}")


if __name__ == "__main__":
    main()
