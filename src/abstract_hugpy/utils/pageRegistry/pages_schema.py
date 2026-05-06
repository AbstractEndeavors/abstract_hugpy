# pages_schema.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal, Tuple

FieldKind = Literal["text", "textarea", "number", "checkbox", "select", "file", "files"]

@dataclass(frozen=True)
class FieldSpec:
    name: str
    label: str
    kind: FieldKind = "text"
    required: bool = False
    default: Any = None
    choices: Tuple[str, ...] = ()   # only used by "select"
    help: str = ""

@dataclass(frozen=True)
class PageSpec:
    key: str                         # url slug, e.g. "summarizer/summarize"
    title: str
    category: str                    # group on index
    endpoint: str                    # absolute path of the JSON route
    method: Literal["POST", "GET"] = "POST"
    fields: Tuple[FieldSpec, ...] = ()
    is_upload: bool = False          # multipart/form-data
    queued: bool = True              # submit via InferenceQueue
    submit_label: str = "Run"
    description: str = ""
