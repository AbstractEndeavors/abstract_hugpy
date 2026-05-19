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
from __future__ import annotations

from typing import AsyncIterator, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict


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
