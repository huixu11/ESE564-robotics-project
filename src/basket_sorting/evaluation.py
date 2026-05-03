from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

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


def save_gif(
    path: str | Path,
    frames: list[np.ndarray],
    duration_ms: int = 60,
    max_frames: int | None = 3000,
    frame_stride: int = 1,
    captions: list[str] | None = None,
) -> int:
    if not frames:
        return 0
    out = Path(path)
    ensure_dir(out.parent)
    frame_stride = max(1, int(frame_stride))
    sampled_frames = frames[::frame_stride]
    sampled_captions = captions[::frame_stride] if captions is not None else [None] * len(sampled_frames)
    if max_frames is not None and max_frames > 0 and len(sampled_frames) > max_frames:
        indices = np.linspace(0, len(sampled_frames) - 1, max_frames, dtype=int)
        sampled_frames = [sampled_frames[int(idx)] for idx in indices]
        sampled_captions = [sampled_captions[int(idx)] for idx in indices]
    pil_frames = [_frame_to_image(frame, caption) for frame, caption in zip(sampled_frames, sampled_captions)]
    pil_frames[0].save(
        out,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return len(sampled_frames)


def _frame_to_image(frame: np.ndarray, caption: str | None) -> Image.Image:
    image = Image.fromarray(frame).convert("RGB")
    if not caption:
        return image
    draw = ImageDraw.Draw(image)
    text = f"Language: {caption}"
    lines = textwrap.wrap(text, width=46) or [text]
    margin = 5
    line_bboxes = [draw.textbbox((margin, margin), line) for line in lines]
    line_height = max((bbox[3] - bbox[1] for bbox in line_bboxes), default=10) + 2
    pad = 4
    draw.rectangle(
        [
            0,
            0,
            image.width,
            margin + pad + len(lines) * line_height,
        ],
        fill=(0, 0, 0),
    )
    for idx, line in enumerate(lines):
        draw.text((margin, margin + idx * line_height), line, fill=(255, 255, 255))
    return image
