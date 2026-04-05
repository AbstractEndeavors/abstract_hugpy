import os,re,unicodedata,bs4,urllib
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

