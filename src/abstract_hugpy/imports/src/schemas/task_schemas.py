from .imports import *
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
