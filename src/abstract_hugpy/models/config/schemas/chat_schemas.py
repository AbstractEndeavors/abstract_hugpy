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
DEFAULT_MAX_TOKENS=4096

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str


# in your schemas module
from typing import Union, Mapping
from uuid import uuid4

ChatInput = Union["ChatRequest", Mapping, str]  # request | dict-ish | bare prompt

class ChatRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    model_key: str
    messages: list[ChatMessage]
    max_new_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    unbounded: bool = False
    max_chunks: Optional[int] = None

    @classmethod
    def coerce(cls, value: ChatInput, *, model_key: Optional[str] = None) -> "ChatRequest":
        # already a request — pass through, no merging, no override
        if isinstance(value, cls):
            return value

        # bare prompt string
        if isinstance(value, str):
            if not model_key:
                raise ValueError("coerce(str) requires model_key")
            return cls(model_key=model_key,
                       messages=[{"role": "user", "content": value}])

        # mapping: validate through the model, no kwarg sprawl
        if isinstance(value, Mapping):
            data = dict(value)
            if "messages" not in data and "prompt" in data:
                prompt = data.pop("prompt")
                system = data.pop("system", None)
                msgs = []
                if system:
                    msgs.append({"role": "system", "content": system})
                msgs.append({"role": "user", "content": prompt})
                data["messages"] = msgs
            if "model_key" not in data and model_key:
                data["model_key"] = model_key
            return cls.model_validate(data)

        raise TypeError(f"cannot coerce {type(value).__name__} to ChatRequest")
class ChatResult(TaskResult):
    text: str
    finish_reason: Literal["stop", "max_tokens", "cancelled", "error"]
    usage: Optional[dict] = None
    output_chunks: int = 0
