from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from basket_sorting.config import ensure_dir


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    successes = np.asarray([r["success"] for r in results], dtype=float)
    steps = np.asarray([r["steps"] for r in results], dtype=float)
    return {
        "episodes": len(results),
        "success_rate": float(successes.mean()) if len(successes) else 0.0,
        "avg_steps": float(steps.mean()) if len(steps) else 0.0,
        "results": results,
    }


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    ensure_dir(out.parent)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_gif(path: str | Path, frames: list[np.ndarray], duration_ms: int = 60) -> None:
    if not frames:
        return
    out = Path(path)
    ensure_dir(out.parent)
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        out,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )
