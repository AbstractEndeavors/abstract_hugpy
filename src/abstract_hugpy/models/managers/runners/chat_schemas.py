"""Chat-family request / result types.

Used by both DeepCoderChatRunner (transformers) and LlamaCppChatRunner
(llama_cpp). One schema, two backends. The schema doesn't know or care
which backend serves it.

Why a separate ChatMessage class instead of {role: str, content: str}:
    Pydantic validation at the boundary catches typos like 'rolle' or
    missing content before they hit the model loader. Cheap defense.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .protocol import TaskRequest, TaskResult


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(TaskRequest):
    """One-shot or streaming chat request.

    Defaults match the runner-level DEFAULT_MAX_TOKENS / etc. so requests
    that omit fields get sensible behavior. The runner clamps these
    further if it has its own ceilings.
    """
    messages: list[ChatMessage] = Field(min_length=1)
    max_new_tokens: int = Field(default=2048, gt=0, le=32768)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    do_sample: bool = False

    # Optional: opt into the auto-continue behavior for very long outputs.
    # Runners that don't implement unbounded mode ignore this.
    unbounded: bool = False


class ChatResult(TaskResult):
    text: str
    finish_reason: Literal["stop", "max_tokens", "cancelled", "error"]
    usage: Optional[dict] = None
    output_chunks: int = 0
