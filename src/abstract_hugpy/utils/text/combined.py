"""High-level text / media analysis utilities.

These functions sit on top of the runner dispatch layer and optionally
require third-party packages (abstract_ocr, abstract_webtools) that may
not be installed in every environment. All such imports are guarded so
that importing this module never raises ImportError.
"""
from __future__ import annotations

import logging
import os.path as osp
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .imports import *

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional third-party deps — fail silently at import time.
# ---------------------------------------------------------------------------

try:
    from abstract_ocr import paddle_image as _paddle_image
except ImportError:
    _paddle_image = None

try:
    from abstract_webtools import (
        get_soup_text as _get_soup_text,
        get_body_from_url as _get_body_from_url,
    )
    from bs4 import BeautifulSoup
except ImportError:
    _get_soup_text = None
    _get_body_from_url = None
    BeautifulSoup = None

try:
    from PyPDF2 import PdfReader as _PdfReader
except ImportError:
    _PdfReader = None

# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


class GenParams(BaseModel):
    model_config = ConfigDict(frozen=True)
    max_new_tokens: int = 100
    temperature: float = 0.6
    top_p: float = 0.95
    use_chat_template: bool = False
    messages: Optional[List[Dict[str, str]]] = None
    do_sample: bool = False
    unbounded: bool = True

    def to_kwargs(self) -> dict:
        return self.model_dump()


@dataclass(frozen=True)
class AnalyzePresets:
    scope: str = "full"
    summary_preset: str = "article"
    keyword_preset: str = "seo"


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def get_num_pdf_pages(pdf_path: str) -> int:
    if _PdfReader is None:
        raise ImportError("PyPDF2 is required. pip install PyPDF2")
    reader = _PdfReader(pdf_path)
    return len(reader.pages)


def extract_single_pdf_page_text(pdf_path: str, page_index: int) -> str:
    if _PdfReader is None:
        raise ImportError("PyPDF2 is required. pip install PyPDF2")
    reader = _PdfReader(pdf_path)
    return reader.pages[page_index].extract_text() or ""


# ---------------------------------------------------------------------------
# Extractor registry
# ---------------------------------------------------------------------------

Extractor = Callable[[str], str]
_EXTRACTORS: dict[str, Extractor] = {}


def register_extractor(kind: str, fn: Extractor) -> None:
    if kind in _EXTRACTORS:
        raise ValueError(f"extractor {kind!r} already registered")
    _EXTRACTORS[kind] = fn


def get_extractor(kind: str) -> Extractor:
    if kind not in _EXTRACTORS:
        raise KeyError(f"unknown extractor {kind!r}; have {sorted(_EXTRACTORS)}")
    return _EXTRACTORS[kind]


def _image_extractor(path: str) -> str:
    if _paddle_image is None:
        raise ImportError("abstract_ocr is required for image extraction")
    return _paddle_image(path)


def _transcribe_extractor(path: str) -> str:
    from ...managers.dispatch import execute_prompt
    result = execute_prompt(file=path, task="automatic-speech-recognition")
    return getattr(result, "text", str(result))


def _website_extractor(url: str) -> str:
    if _get_soup_text is None:
        raise ImportError("abstract_webtools is required for website extraction")
    return _get_soup_text(url)


register_extractor("image", _image_extractor)
register_extractor("audio", _transcribe_extractor)
register_extractor("video", _transcribe_extractor)
register_extractor("website", _website_extractor)


def _pdf_full_text(path: str) -> str:
    parts = []
    for i in range(get_num_pdf_pages(path)):
        parts.append(extract_single_pdf_page_text(pdf_path=path, page_index=i))
    return "\n\n".join(parts)


register_extractor("pdf", _pdf_full_text)


def source_to_text(source: str, kind: str | None = None) -> str:
    if kind in (None, "", "text"):
        return source
    return get_extractor(kind)(source)


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def summarize(source: str, kind: str = None,
              presets: AnalyzePresets = AnalyzePresets()) -> dict:
    from ...utils.seo.pdf_utils import _analyze
    text = source_to_text(source, kind)
    report = _analyze(
        text,
        scope=presets.scope,
        summary_preset=presets.summary_preset,
        keyword_preset=presets.keyword_preset,
    )
    return report.to_dict()


async def analyze(
    source: str,
    kind: str = "text",
    prompt: str = "Please analyze the following content.",
    params: GenParams | None = None,
    model_key: str = None,
) -> Any:
    from ...managers.dispatch import runner_for
    from ...imports.src.schemas.chat_schemas import ChatRequest
    from ...imports.src.constants import DEFAULT_CHAT_MODEL
    model_key = model_key or DEFAULT_CHAT_MODEL
    params = params or GenParams()
    text = source_to_text(source, kind)
    params = params.model_copy(update={
        "messages": [{"role": "user", "content": f"{prompt}\n\n{text}"}]
    })
    payload = params.model_dump(exclude={"use_chat_template"})
    req = ChatRequest(model_key=model_key, **payload)
    runner = runner_for(model_key)
    return await runner.run(req)


# ---------------------------------------------------------------------------
# Back-compat shims
# ---------------------------------------------------------------------------

_GEN_KEYS = {"max_new_tokens", "temperature", "top_p", "use_chat_template",
             "messages", "do_sample"}


def _filter_gen_kw(kw: dict) -> dict:
    return {k: v for k, v in kw.items() if k in _GEN_KEYS}


def image_analysis(path: str, prompt: str = "Please describe the image",
                   max_new_tokens: int = 100):
    from ...managers.vision.vision_coder import deepcoder_image_analysis
    return deepcoder_image_analysis(image_path=path, prompt=prompt,
                                    max_new_tokens=max_new_tokens)


def image_to_text(path: str) -> str:
    return get_extractor("image")(path)


def video_to_text(path: str) -> str:
    return get_extractor("video")(path)


def audio_to_text(path: str) -> str:
    return get_extractor("audio")(path)


def website_to_text(url: str) -> str:
    return _website_extractor(url)
