from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


SPEED_VALUES = {
    "careful": 0.0,
    "normal": 0.5,
    "fast": 1.0,
}


@dataclass(frozen=True)
class TaskSpec:
    instruction: str
    target_object: str
    target_basket: str
    speed_name: str
    speed: float
    subtasks: tuple[str, ...]


def _contains_any(text: str, candidates: Iterable[str]) -> bool:
    return any(candidate in text for candidate in candidates)


def parse_instruction(instruction: str) -> TaskSpec:
    """Parse the restricted language commands used by the baseline."""

    text = instruction.lower().strip()
    if _contains_any(text, ["cracker", "box"]):
        target_object = "cracker_box"
    elif _contains_any(text, ["mustard", "bottle"]):
        target_object = "mustard_bottle"
    else:
        raise ValueError(f"Could not infer target object from instruction: {instruction!r}")

    if "left" in text:
        target_basket = "left"
    elif "right" in text:
        target_basket = "right"
    else:
        raise ValueError(f"Could not infer target basket from instruction: {instruction!r}")

    speed_name = "normal"
    for name in SPEED_VALUES:
        if name in text:
            speed_name = name
            break

    subtasks = (
        f"approach {target_object}",
        f"grasp {target_object}",
        f"place in {target_basket} basket",
    )
    return TaskSpec(
        instruction=instruction,
        target_object=target_object,
        target_basket=target_basket,
        speed_name=speed_name,
        speed=SPEED_VALUES[speed_name],
        subtasks=subtasks,
    )


def make_instruction(target_object: str, target_basket: str, speed_name: str = "normal") -> str:
    object_text = {
        "cracker_box": "cracker box",
        "mustard_bottle": "mustard bottle",
    }.get(target_object, target_object.replace("_", " "))
    prefix = "" if speed_name == "normal" else f"{speed_name} "
    return f"{prefix}place the {object_text} in the {target_basket} basket"
