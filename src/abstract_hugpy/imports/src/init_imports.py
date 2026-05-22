from __future__ import annotations
import os,re,unicodedata,bs4,urllib,tempfile,copy
import os.path as osp
from pydantic import BaseModel, ConfigDict, Field
from dataclasses import dataclass, asdict, field
from abstract_utilities import (
    SingletonMeta,
    make_list,
    get_any_value,
    get_logFile,
    safe_read_from_json,
    get_env_value
    )
from typing import (
    List,
    Optional,
    Callable,
    Dict,
    Tuple,
    Union,
    Literal
    )
from urllib.parse import (
    urlunparse,
    unquote,
    quote,
    urlparse,
    parse_qs
    )
from collections import Counter
from typing import *
from PyPDF2 import PdfReader
from uuid import uuid1
from pathlib import Path
from abstract_utilities import *
from abstract_webtools import requests,derive_approved_headers_user_agent_session_for_url
from datetime import datetime, timezone


