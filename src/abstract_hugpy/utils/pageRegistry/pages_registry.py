# pages_registry.py
from typing import Dict, List
from .pages_schema import PageSpec

_PAGES: Dict[str, PageSpec] = {}

def register_page(spec: PageSpec) -> None:
    if spec.key in _PAGES:
        raise KeyError(f"Page {spec.key!r} already registered")
    _PAGES[spec.key] = spec

def get_page(key: str) -> PageSpec:
    if key not in _PAGES:
        raise KeyError(f"Unknown page {key!r}")
    return _PAGES[key]

def pages_by_category() -> Dict[str, List[PageSpec]]:
    out: Dict[str, List[PageSpec]] = {}
    for p in sorted(_PAGES.values(), key=lambda x: (x.category, x.title)):
        out.setdefault(p.category, []).append(p)
    return out
