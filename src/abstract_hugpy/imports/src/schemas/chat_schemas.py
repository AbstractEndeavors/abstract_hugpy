from .imports import *
from .task_schemas import *
ChatInput = Union["ChatRequest", Mapping, str]  # request | dict-ish | bare prompt

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: ROLES = "user"
    content: str

class ChatRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    request_id: str = Field(default_factory=lambda: get_request_id())
    model_key: str = None
    messages: list[ChatMessage]
    max_new_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    do_sample: bool = False
    unbounded: bool = False
    max_chunks: Optional[int] = None
    file: Optional[str] = None
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
                file = data.pop("file")
                if file:
                    content = read_from_file(file)
                    prompt = f"{prompt}\n------{file}------\n{content}"
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
