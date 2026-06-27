from .imports import *
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
