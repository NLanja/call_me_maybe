"""Helpers for displaying function-calling results in the terminal."""


import json
from .config_parse import FunctionCallResult


def display_result(
    index: int, total: int, result: FunctionCallResult
) -> None:
    """Print a single function-calling result to the terminal.

    Args:
        index: 1-based position of this result among all prompts.
        total: Total number of prompts being processed.
        result: The function-calling result to display.
    """
    print(f"\n[{index}/{total}] {result.prompt!r}")
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    print("-" * 40)
