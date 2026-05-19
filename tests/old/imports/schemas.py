from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"] = "user"
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    request_id: str = Field(min_length=1)
    model_key: str = "qwen25_coder_15b_gguf"

    messages: list[ChatMessage] = Field(min_length=1)

    max_new_tokens: int = Field(default=512, gt=0, le=4096)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    do_sample: bool = False


class TokenEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["token"] = "token"
    request_id: str
    text: str


class DoneEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["done"] = "done"
    request_id: str
    input_tokens: int
    output_chunks: int
    finish_reason: Literal["stop", "max_tokens", "cancelled", "error"]


class ErrorEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["error"] = "error"
    request_id: str
    message: str


StreamEvent = Union[TokenEvent, DoneEvent, ErrorEvent]
