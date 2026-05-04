from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()

from basket_sorting.config import load_config
from basket_sorting.env_factory import make_env
from basket_sorting.evaluation import save_json
from basket_sorting.perception import estimate_color_positions


def _metric_summary(errors_m: list[float]) -> dict[str, float]:
    if not errors_m:
        return {
            "count": 0,
            "mean_error_m": 0.0,
            "mean_error_cm": 0.0,
            "max_error_m": 0.0,
            "max_error_cm": 0.0,
        }
    values = np.asarray(errors_m, dtype=float)
    return {
        "count": int(values.size),
        "mean_error_m": float(values.mean()),
        "mean_error_cm": float(values.mean() * 100.0),
        "max_error_m": float(values.max()),
        "max_error_cm": float(values.max() * 100.0),
    }


def _perception_rgb(env: Any, obs: dict[str, Any]) -> np.ndarray:
    if hasattr(env, "names") and env.names.perception_camera_id != env.names.camera_id:
        return env._render_camera(env.names.perception_camera_id)
    return np.asarray(obs["rgb"], dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure color-perception tabletop localization error.")
    parser.add_argument("--config", default="configs/class_panda.yaml")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=900)
    parser.add_argument("--out", default="runs/perception_sanity.json")
    args = parser.parse_args()

    config = load_config(args.config)
    perception_cfg = config["env"].get("perception", {})
    if not perception_cfg.get("enabled", False):
        raise RuntimeError(f"Perception is disabled in {args.config}.")

    object_names = list(config["env"]["object_names"])
    errors_by_object: dict[str, list[float]] = {name: [] for name in object_names}
    misses_by_object: dict[str, int] = {name: 0 for name in object_names}
    records: list[dict[str, Any]] = []

    env = make_env(config, seed=args.seed)
    try:
        for sample_idx in range(args.samples):
            seed = args.seed + sample_idx
            obs = env.reset(seed=seed)
            estimates = estimate_color_positions(_perception_rgb(env, obs), perception_cfg)
            for name in object_names:
                true_xy = np.asarray(obs["sim_objects"][name][:2], dtype=float)
                if name not in estimates:
                    misses_by_object[name] += 1
                    records.append(
                        {
                            "sample": sample_idx,
                            "seed": seed,
                            "object": name,
                            "detected": False,
                            "true_xy": true_xy.tolist(),
                        }
                    )
                    continue
                estimate_xy = np.asarray(estimates[name][:2], dtype=float)
                error_m = float(np.linalg.norm(estimate_xy - true_xy))
                errors_by_object[name].append(error_m)
                records.append(
                    {
                        "sample": sample_idx,
                        "seed": seed,
                        "object": name,
                        "detected": True,
                        "true_xy": true_xy.tolist(),
                        "estimate_xy": estimate_xy.tolist(),
                        "error_m": error_m,
                        "error_cm": error_m * 100.0,
                    }
                )
    finally:
        env.close()

    all_errors = [error for errors in errors_by_object.values() for error in errors]
    total_possible = int(args.samples * len(object_names))
    total_detected = len(all_errors)
    payload = {
        "config": args.config,
        "samples": int(args.samples),
        "seed": int(args.seed),
        "objects": {
            name: {
                **_metric_summary(errors_by_object[name]),
                "misses": int(misses_by_object[name]),
                "detection_rate": float(len(errors_by_object[name]) / max(1, args.samples)),
            }
            for name in object_names
        },
        "overall": {
            **_metric_summary(all_errors),
            "total_possible": total_possible,
            "total_detected": total_detected,
            "misses": int(total_possible - total_detected),
            "detection_rate": float(total_detected / max(1, total_possible)),
        },
        "records": records,
    }
    save_json(Path(args.out), payload)
    overall = payload["overall"]
    print(
        "perception_sanity "
        f"config={args.config} samples={args.samples} "
        f"detection_rate={overall['detection_rate']:.3f} "
        f"mean_error_cm={overall['mean_error_cm']:.2f} "
        f"max_error_cm={overall['max_error_cm']:.2f} "
        f"out={args.out}"
    )


if __name__ == "__main__":
    main()
