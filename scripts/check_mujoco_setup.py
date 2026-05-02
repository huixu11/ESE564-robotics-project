from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from basket_sorting.config import load_config
from basket_sorting.env_factory import make_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate configured MuJoCo assets and names.")
    parser.add_argument("--config", default="configs/mujoco_template.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    env = make_env(config)
    obs = env.reset(seed=config.get("seed", 0))
    print("mujoco_setup_ok=True")
    print(f"instruction={obs['task'].instruction!r}")
    print(f"ee_pos={obs['ee_pos']}")
    print(f"objects={obs['objects']}")


if __name__ == "__main__":
    main()
