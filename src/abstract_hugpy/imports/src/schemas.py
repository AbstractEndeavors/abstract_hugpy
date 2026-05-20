from __future__ import annotations
"""Chat-family request / result types.

Used by both DeepCoderChatRunner (transformers) and LlamaCppChatRunner
(llama_cpp). One schema, two backends. The schema doesn't know or care
which backend serves it.

Why a separate ChatMessage class instead of {role: str, content: str}:
    Pydantic validation at the boundary catches typos like 'rolle' or
    missing content before they hit the model loader. Cheap defense.
"""
"""Shared types for the runner protocol.

Every task family (chat, summarize, transcribe, vision, keyword, ...) defines:
    - a TaskRequest subclass describing its inputs
    - a TaskResult subclass describing its output
    - a Runner class implementing .run() and optionally .stream()

The route layer doesn't import any concrete runner — it goes through
runner_for(model_key) and operates on the Runner protocol only.

Naming:
    TaskRequest / TaskResult are deliberately not called BaseRequest /
    BaseResult so they don't collide with the dozen other "Base*" things
    that already exist in this codebase.

Streaming events:
    Reuses the StreamEvent / TokenEvent / DoneEvent / ErrorEvent types
    from the existing schema. Don't redefine them here — that's how you
    end up with two parallel event hierarchies.
"""
from typing import (
    Literal,
    Optional,
    Union,
    Mapping,
    Any,
    AsyncIterator,
    Protocol,
    runtime_checkable
    )

# in your schemas module
from pydantic import  (
    BaseModel,
    ConfigDict,
    Field,
    field_validator
    )
from .constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    FINISH_REASONS,
    ROLES
    )
from .utils import (
    get_request_id,
    get_messages,
    get_message
    )
from .init_imports import dataclass, asdict
# ---------------------------------------------------------------------------
# Request / Result base shapes
# ---------------------------------------------------------------------------

class TaskRequest(BaseModel):
    """Marker base for all task-family request schemas.

    request_id and model_key are universal — every request we route needs
    to identify itself and say which model to send to. Everything else
    (messages, audio_path, prompt, etc.) lives on the subclass.
    """
    model_config = ConfigDict(extra="forbid")
    request_id: str
    model_key: str


class TaskResult(BaseModel):
    """Marker base for all task-family result schemas.

    `ok` and `error` are part of the base so the route layer can return
    a consistent envelope regardless of which runner produced the result.
    Successful runs leave `error=None`; failures set ok=False and put the
    message in error.
    """
    model_config = ConfigDict(extra="allow")
    request_id: str
    model_key: str
    ok: bool = True
    error: Optional[str] = None

ChatInput = Union["ChatRequest", Mapping, str]  # request | dict-ish | bare prompt

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: ROLES = "user"
    content: str

class ChatRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    request_id: str = Field(default_factory=lambda: get_request_id())
    model_key: str = DEFAULT_MODEL
    messages: list[ChatMessage]
    max_new_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    do_sample: bool = False
    unbounded: bool = False
    max_chunks: Optional[int] = None

    @field_validator("messages", mode="before")
    @classmethod
    def normalize_messages(cls, value: Any) -> Any:
        if isinstance(value, str):
            return get_messages(value)
        return value

    @classmethod
    def coerce(cls, value: ChatInput, *, model_key: Optional[str] = None) -> "ChatRequest":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(model_key=model_key, messages=value)  # validator handles it
        if isinstance(value, Mapping):
            data = dict(value)
            if "messages" not in data and "prompt" in data:
                prompt = data.pop("prompt")
                system = data.pop("system", None)
                msgs = []
                if system:
                    msg = get_message(content=prompt,role="system")
                    msgs.append(msg)
                msg = get_message(content=prompt,role="user")
                msgs.append(msg)
                data["messages"] = msgs
            if "model_key" not in data and model_key:
                data["model_key"] = model_key
            return cls.model_validate(data)
        raise TypeError(f"cannot coerce {type(value).__name__} to ChatRequest")

class ChatResult(TaskResult):
    text: str
    finish_reason: FINISH_REASONS
    usage: Optional[dict] = None
    output_chunks: int = 0
    
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
    finish_reason: FINISH_REASONS

class ErrorEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["error"] = "error"
    request_id: str
    message: str


# ---------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    name: str
    hub_id: str
    folder: str
    task: str
    framework: str = "transformers"
    filename: Optional[str] = None
    include: Optional[str] = None
    model_max_length: Optional[str] = DEFAULT_MAX_TOKENS
    port: Optional[int] = None
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeepCoderRuntime:
    model_dir: str
    device: str
    torch_dtype: Any
    use_quantization: bool = False
    use_flash_attention: bool = False
    local_files_only: bool = True
    max_new_tokens_cap: int = DEFAULT_MAX_TOKENS
    max_concurrent_generations: int = 1

    def cache_key(self) -> tuple:
        return (
            self.model_dir,
            self.device,
            str(self.torch_dtype),
            self.use_quantization,
            self.use_flash_attention,
            self.local_files_only,
            self.max_new_tokens_cap,
            self.max_concurrent_generations,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_dir": self.model_dir,
            "device": self.device,
            "torch_dtype": safe_dtype_name(self.torch_dtype),
            "use_quantization": self.use_quantization,
            "use_flash_attention": self.use_flash_attention,
            "local_files_only": self.local_files_only,
            "max_new_tokens_cap": self.max_new_tokens_cap,
            "max_concurrent_generations": self.max_concurrent_generations,
        }


# ---------------------------------------------------------------------------
# Runner protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Runner(Protocol):
    """The contract every runner implements.

    request_type / result_type are class attributes (not instance attrs)
    so the dispatch layer can ask 'what shape does this runner expect?'
    without instantiating anything.

    .run() is required. .stream() is optional — runners that don't support
    streaming should raise NotImplementedError so the route layer can
    return a clean 4xx instead of a 500.
    """

    request_type: type[TaskRequest]
    result_type: type[TaskResult]
    model_key: str

    async def run(self, req: TaskRequest) -> TaskResult: ...

    async def stream(
        self,
        req: TaskRequest,
        cancel_event,
    ) -> AsyncIterator: ...
    
StreamEvent = Union[TokenEvent, DoneEvent, ErrorEvent]
