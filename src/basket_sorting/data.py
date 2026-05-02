from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from basket_sorting.config import ensure_dir
from basket_sorting.features import bc_feature_vector


class DemoWriter:
    def __init__(self, root: str | Path, save_images: bool = True, image_format: str = "png") -> None:
        self.root = ensure_dir(root)
        self.episodes_dir = ensure_dir(self.root / "episodes")
        self.images_dir = ensure_dir(self.root / "images")
        self.metadata_path = self.root / "metadata.jsonl"
        self.save_images = save_images
        self.image_format = image_format

    def write_episode(self, episode_id: int, instruction: str, records: list[dict[str, Any]], success: bool) -> Path:
        state_features = np.asarray([r["state_features"] for r in records], dtype=np.float32)
        actions = np.asarray([r["action"] for r in records], dtype=np.float32)
        speeds = np.asarray([r["speed"] for r in records], dtype=np.float32)
        phases = np.asarray([r["phase"] for r in records])

        image_paths: list[str] = []
        if self.save_images:
            episode_image_dir = ensure_dir(self.images_dir / f"episode_{episode_id:05d}")
            for idx, record in enumerate(records):
                rel_path = Path("images") / f"episode_{episode_id:05d}" / f"{idx:04d}.{self.image_format}"
                Image.fromarray(record["rgb"]).save(self.root / rel_path)
                image_paths.append(str(rel_path))

        out_path = self.episodes_dir / f"episode_{episode_id:05d}.npz"
        np.savez_compressed(
            out_path,
            state_features=state_features,
            actions=actions,
            speeds=speeds,
            phases=phases,
            image_paths=np.asarray(image_paths),
        )
        metadata = {
            "episode_id": episode_id,
            "instruction": instruction,
            "success": bool(success),
            "steps": len(records),
            "episode_path": str(out_path.relative_to(self.root)),
        }
        with self.metadata_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metadata) + "\n")
        return out_path


def load_demo_arrays(root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    root_path = Path(root)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for episode_path in sorted((root_path / "episodes").glob("*.npz")):
        data = np.load(episode_path, allow_pickle=True)
        state_features = np.asarray(data["state_features"], dtype=np.float32)
        phases = [str(phase) for phase in data["phases"]]
        xs.append(np.asarray([bc_feature_vector(state, phase) for state, phase in zip(state_features, phases)]))
        ys.append(np.asarray(data["actions"], dtype=np.float32))
    if not xs:
        raise FileNotFoundError(f"No demo episodes found under {root_path / 'episodes'}")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)
