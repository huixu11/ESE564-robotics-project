from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_COLOR_MASKS = {
    "cracker_box": {
        "lower": [120, 0, 0],
        "upper": [255, 120, 120],
        "min_saturation": 40,
        "dominant_channel": 0,
        "min_channel_margin": 35,
    },
    "mustard_bottle": {"lower": [120, 80, 0], "upper": [255, 230, 140], "min_saturation": 40},
    "left": {"lower": [40, 70, 80], "upper": [150, 190, 210]},
    "right": {"lower": [40, 80, 50], "upper": [150, 190, 150]},
}


def estimate_color_positions(
    rgb: np.ndarray,
    perception_cfg: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    cfg = perception_cfg or {}
    affine = np.asarray(
        cfg.get(
            "image_to_world_affine",
            [
                [0.0017647, -0.00378195, 0.60149898],
                [-0.00268734, -0.00116214, 0.62269681],
            ],
        ),
        dtype=float,
    )
    min_pixels = int(cfg.get("min_pixels", 20))
    masks_cfg = cfg.get("color_masks", DEFAULT_COLOR_MASKS)

    image = np.asarray(rgb, dtype=np.uint8)
    estimates: dict[str, np.ndarray] = {}
    used_masks: list[np.ndarray] = []
    for name, mask_cfg in masks_cfg.items():
        lower = np.asarray(mask_cfg["lower"], dtype=np.uint8)
        upper = np.asarray(mask_cfg["upper"], dtype=np.uint8)
        mask = np.all((image >= lower) & (image <= upper), axis=2)
        if "min_saturation" in mask_cfg:
            channel_range = image.max(axis=2).astype(int) - image.min(axis=2).astype(int)
            mask = mask & (channel_range >= int(mask_cfg["min_saturation"]))
        if "dominant_channel" in mask_cfg:
            channel = int(mask_cfg["dominant_channel"])
            margin = int(mask_cfg.get("min_channel_margin", 0))
            others = [idx for idx in range(3) if idx != channel]
            dominant = image[:, :, channel].astype(int)
            strongest_other = np.maximum(image[:, :, others[0]].astype(int), image[:, :, others[1]].astype(int))
            mask = mask & (dominant >= strongest_other + margin)
        if name == "mustard_bottle" and used_masks:
            # Red highlights on the cracker can otherwise leak into the yellow mask.
            mask = mask & ~np.logical_or.reduce(used_masks)
        if name == "cracker_box":
            used_masks.append(mask)
        ys, xs = np.where(mask)
        if len(xs) < min_pixels:
            continue
        pixel = np.array([float(xs.mean()), float(ys.mean()), 1.0], dtype=float)
        xy = affine @ pixel
        estimates[name] = np.array([xy[0], xy[1], 0.04], dtype=float)
    return estimates
