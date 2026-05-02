from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()

from basket_sorting.config import ensure_dir, load_config
from basket_sorting.data import load_demo_arrays
from basket_sorting.policies import train_linear_bc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train lightweight state-feature BC policy.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default="data/demos")
    parser.add_argument("--out", default="models/state_linear_bc.npz")
    args = parser.parse_args()

    config = load_config(args.config)
    x, y = load_demo_arrays(args.data)
    model = train_linear_bc(x, y, ridge_lambda=float(config["training"]["ridge_lambda"]))
    out = Path(args.out)
    ensure_dir(out.parent)
    np.savez_compressed(out, **model)
    print(f"samples={len(x)} train_mse={float(model['train_mse']):.8f} model={out}")


if __name__ == "__main__":
    main()
