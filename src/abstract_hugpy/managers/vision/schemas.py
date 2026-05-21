# vision_schemas.py
import os.path as osp
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator
from .imports import VISION_HOST,DEFAULT_TIMEOUT,DEFAULT_MAX_TOKENS

_BAD_PATH_STRINGS = frozenset({
    "", "[object object]", "undefined", "null", "none",
})


class VisionBackendConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_key: str = Field(min_length=1)
    # explicit: None => in-process; int => http client to host:port
    port: Optional[int] = Field(default=None, gt=0, le=65535)
    host: str = VISION_HOST
    timeout_s: float = Field(default=DEFAULT_TIMEOUT, gt=0)
    request_id: str = Field(min_length=1)
    image_path: str
    prompt: str = "Analyze this image."
    max_new_tokens: int = Field(default=DEFAULT_MAX_TOKENS, gt=0, le=4096)
    # image-token budget; None => use the coder's configured cfg.max_tokens
    max_tokens: Optional[int] = Field(default=None, gt=0)

    @field_validator("image_path")
    @classmethod
    def _validate_image_path(cls, v: str) -> str:
        if not isinstance(v, str):
            raise TypeError(
                f"image_path must be a string, got {type(v).__name__}: {v!r}"
            )
        cleaned = v.strip()
        if cleaned.lower() in _BAD_PATH_STRINGS:
            raise ValueError(
                f"image_path looks like a serialization artifact, not a real path: {v!r}"
            )
        if not osp.exists(cleaned):
            raise FileNotFoundError(f"Image not found: {cleaned}")
        return cleaned


class VisionRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str = Field(min_length=1)
    model_key: str = Field(min_length=1)
    image_path: str
    prompt: str = "Analyze this image."
    max_new_tokens: int = Field(default=128, gt=0, le=4096)
    # image-token budget; None => use the coder's configured cfg.max_tokens
    max_tokens: Optional[int] = Field(default=None, gt=0)

    @field_validator("image_path")
    @classmethod
    def _validate_image_path(cls, v: str) -> str:
        if not isinstance(v, str):
            raise TypeError(
                f"image_path must be a string, got {type(v).__name__}: {v!r}"
            )
        cleaned = v.strip()
        if cleaned.lower() in _BAD_PATH_STRINGS:
            raise ValueError(
                f"image_path looks like a serialization artifact, not a real path: {v!r}"
            )
        if not osp.exists(cleaned):
            raise FileNotFoundError(f"Image not found: {cleaned}")
        return cleaned

class VisionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str = Field(min_length=1)
    model_key: str = Field(min_length=1)
    text: str
    # optional so the same envelope can carry a failure back through the queue;
    # if you'd rather split, make a sibling VisionError model and union at the consumer
    error: Optional[str] = None
