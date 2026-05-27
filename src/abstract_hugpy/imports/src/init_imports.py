from __future__ import annotations

import base64
import bs4
import copy
import glob
import json
import os
import os.path as osp
import re
import tempfile
import unicodedata
import urllib
import uuid
from collections import Counter
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)
from typing import *
from urllib.parse import urlunparse, unquote, quote, urlparse, parse_qs
from uuid import uuid1

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ._compat import (
    SingletonMeta,
    make_list,
    get_any_value,
    get_logFile,
    safe_read_from_json,
    safe_load_from_json,
    get_env_value,
    requests,
    derive_approved_headers_user_agent_session_for_url,
)

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None


def read_from_file(path: str) -> str:
    """Read a text file and return its contents."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def derive_media_type(path: str) -> str:
    """Guess media type from file extension."""
    ext = os.path.splitext(path)[-1].lower()
    _MAP = {
        ".mp4": "video", ".avi": "video", ".mov": "video", ".mkv": "video",
        ".mp3": "audio", ".wav": "audio", ".m4a": "audio", ".ogg": "audio", ".flac": "audio",
        ".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image", ".webp": "image",
        ".pdf": "document",
        ".py": "code", ".js": "code", ".ts": "code", ".cpp": "code", ".c": "code",
        ".txt": "text", ".md": "text",
    }
    return _MAP.get(ext, "text")
