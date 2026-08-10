"""Tool definitions the agent exposes to the LLM, plus their executor.

Schemas follow the OpenAI function-calling format, which Nebius AI Studio's
chat completions endpoint supports.
"""

from __future__ import annotations

import json

from .grid import Grid, Puzzle

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_state",
            "description": "Get the current grid, all clues, and per-slot fill patterns "
            "('.' = unknown letter).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fill_slot",
            "description": "Write an answer into a slot. Fails with a conflict list if it "
            "contradicts crossing letters (pass overwrite=true to force).",
            "parameters": {
                "type": "object",
                "properties": {
                    "slot": {"type": "string", "description": "Slot id, e.g. '4A' or '2D'"},
                    "word": {"type": "string", "description": "Answer, letters only"},
                    "overwrite": {"type": "boolean", "default": False},
                },
                "required": ["slot", "word"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clear_slot",
            "description": "Blank out a slot (crossing entries lose the shared letter).",
            "parameters": {
                "type": "object",
                "properties": {"slot": {"type": "string"}},
                "required": ["slot"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit the grid as final. Call once every slot is filled and "
            "consistent.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


class ToolExecutor:
    """Executes tool calls against a live Grid. One instance per solve."""

    def __init__(self, puzzle: Puzzle):
        self.puzzle = puzzle
        self.grid: Grid = puzzle.make_grid()
        self.submitted = False

    def execute(self, name: str, args: dict) -> str:
        try:
            handler = getattr(self, f"_tool_{name}", None)
            if handler is None:
                return json.dumps({"error": f"unknown tool {name!r}"})
            return json.dumps(handler(**args))
        except (KeyError, ValueError, TypeError) as exc:
            return json.dumps({"error": str(exc)})

    def _tool_get_state(self) -> dict:
        return {
            "title": self.puzzle.title,
            "grid": self.grid.render(),
            "slots": [
                {
                    "slot": slot.id,
                    "clue": slot.clue,
                    "length": slot.length,
                    "pattern": self.grid.slot_pattern(slot.id),
                }
                for slot in self.grid.slots.values()
            ],
        }

    def _tool_fill_slot(self, slot: str, word: str, overwrite: bool = False) -> dict:
        conflicts = self.grid.fill_slot(slot, word, overwrite=overwrite)
        if conflicts and not overwrite:
            return {"ok": False, "conflicts": conflicts}
        return {"ok": True, "pattern": self.grid.slot_pattern(slot), "overwrote": conflicts}

    def _tool_clear_slot(self, slot: str) -> dict:
        self.grid.clear_slot(slot)
        return {"ok": True}

    def _tool_submit(self) -> dict:
        self.submitted = True
        return {"ok": True, "complete": self.grid.is_complete()}
