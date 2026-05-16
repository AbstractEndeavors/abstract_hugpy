from __future__ import annotations

from typing import Any


def message_to_dict(message: Any) -> dict:
    if hasattr(message, "model_dump"):
        return message.model_dump()

    if isinstance(message, dict):
        return {
            "role": str(message.get("role", "user")),
            "content": str(message.get("content", "")),
        }

    return {
        "role": str(getattr(message, "role", "user")),
        "content": str(getattr(message, "content", "")),
    }


def messages_to_dicts(messages: list[Any]) -> list[dict]:
    return [message_to_dict(message) for message in messages]
