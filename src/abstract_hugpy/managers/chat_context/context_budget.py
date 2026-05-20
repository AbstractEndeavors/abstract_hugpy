from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
from .imports import DEFAULT_MAX_TOKENS

@dataclass(frozen=True)
class ContextBudget:
    max_context_tokens: int
    reserved_output_tokens: int
    reserved_system_tokens: int = DEFAULT_MAX_TOKENS
    chars_per_token: float = 4.0

    @property
    def input_token_budget(self) -> int:
        return max(
            512,
            self.max_context_tokens
            - self.reserved_output_tokens
            - self.reserved_system_tokens,
        )


def estimate_tokens(text: str, *, chars_per_token: float = 4.0) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / chars_per_token))


def estimate_message_tokens(message: dict, *, chars_per_token: float = 4.0) -> int:
    role = str(message.get("role", "user"))
    content = str(message.get("content", ""))

    # Approximate role/template overhead.
    return (
        estimate_tokens(role, chars_per_token=chars_per_token)
        + estimate_tokens(content, chars_per_token=chars_per_token)
        + 8
    )


def compact_messages_to_budget(
    messages: Iterable[dict],
    budget: ContextBudget,
) -> list[dict]:
    """
    Keep system messages and the newest user/assistant turns that fit.

    This is intentionally extractive. It never invents summaries or rewrites
    prior turns. Add rolling summaries later as a second layer.
    """
    normalized = [
        {
            "role": str(message.get("role", "user")),
            "content": str(message.get("content", "")),
        }
        for message in messages
        if str(message.get("content", "")).strip()
    ]

    system_messages = [m for m in normalized if m["role"] == "system"]
    dialogue_messages = [m for m in normalized if m["role"] != "system"]

    system_cost = sum(
        estimate_message_tokens(m, chars_per_token=budget.chars_per_token)
        for m in system_messages
    )

    remaining = max(256, budget.input_token_budget - system_cost)

    kept_reversed: list[dict] = []
    used = 0

    for message in reversed(dialogue_messages):
        cost = estimate_message_tokens(
            message,
            chars_per_token=budget.chars_per_token,
        )

        if used + cost > remaining:
            break

        kept_reversed.append(message)
        used += cost

    return system_messages + list(reversed(kept_reversed))
