from __future__ import annotations

import logging

from .imports import default_context_tokens_for_model, message_to_dict, DEFAULT_MAX_TOKENS
from .context_budget import ContextBudget, compact_messages_to_budget

logger = logging.getLogger(__name__)


def compact_chat_request(req):
    max_context_tokens = default_context_tokens_for_model(req.model_key)
    requested_output_tokens = req.max_new_tokens or DEFAULT_MAX_TOKENS

    reserved_output_tokens = min(
        requested_output_tokens,
        max(4096, max_context_tokens // 3),
    )

    budget = ContextBudget(
        max_context_tokens=max_context_tokens,
        reserved_output_tokens=reserved_output_tokens,
    )

    raw_messages = [message_to_dict(message) for message in req.messages]

    logger.info(
        "compact_chat_request before: model=%s count=%s roles=%s chars=%s",
        req.model_key,
        len(raw_messages),
        [m.get("role") for m in raw_messages],
        [len(str(m.get("content", ""))) for m in raw_messages],
    )

    compacted_dicts = compact_messages_to_budget(raw_messages, budget)

    logger.info(
        "compact_chat_request after: model=%s count=%s roles=%s chars=%s",
        req.model_key,
        len(compacted_dicts),
        [m.get("role") for m in compacted_dicts],
        [len(str(m.get("content", ""))) for m in compacted_dicts],
    )

    if req.messages:
        message_type = type(req.messages[0])
        compacted_messages = [message_type(**message) for message in compacted_dicts]
    else:
        compacted_messages = []

    return req.model_copy(update={"messages": compacted_messages})
