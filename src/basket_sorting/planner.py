from __future__ import annotations

from .tasks import TaskSpec, parse_instruction


class RuleBasedPlanner:
    """Deterministic planner used before adding a GPT wrapper."""

    def plan(self, instruction: str) -> TaskSpec:
        return parse_instruction(instruction)


class GPTPlanner:
    """Placeholder for the optional GPT-4o planner integration."""

    def __init__(self, model: str = "gpt-4o") -> None:
        self.model = model

    def plan(self, instruction: str) -> TaskSpec:
        raise RuntimeError(
            "GPTPlanner is not wired yet. Use RuleBasedPlanner for the runnable baseline, "
            "then map GPT output into TaskSpec."
        )
