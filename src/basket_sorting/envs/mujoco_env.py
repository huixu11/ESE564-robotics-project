from __future__ import annotations

from typing import Any


class MujocoBasketSortingEnv:
    """Integration placeholder for the class Panda MuJoCo scene."""

    def __init__(self, config: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        try:
            import mujoco  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "MuJoCo is not installed in this environment. Use env.name=kinematic "
                "for the runnable baseline, or install MuJoCo and add the class Panda assets."
            ) from exc
        raise NotImplementedError(
            "Copy the class Panda model/homework IK wrapper into assets/ and implement this "
            "adapter with the same API as KinematicBasketSortingEnv."
        )
