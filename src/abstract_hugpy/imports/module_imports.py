"""
Lazy import accessors for heavy third-party libraries.

Rules:
    1. Every accessor is named `get_<package>`.  Downstream modules that
       re-export a *different* thing must choose a different name
       (e.g. `get_keybert_instance`) so the import accessor is never shadowed.
    2. Accessors that support sub-module lookup (`module` param) build a
       dispatch dict once per call.  The dict is cheap; the import behind it
       is cached by Python's import system.
    3. Torch is guarded against the _refs/_prims recursion bug in PyTorch 2.x
       by disabling dynamo eagerly.
    4. `require()` is the gateway for any code that *cannot* function without
       a dependency.  It turns a silent nullProxy into an immediate, readable
       ImportError — right at init, not four calls later as a TypeError.
"""

from __future__ import annotations

from abstract_utilities import lazy_import
from abstract_utilities.import_utils.src.layze_import_utils.nullProxy import nullProxy


# ---------------------------------------------------------------------------
# Dependency gate
# ---------------------------------------------------------------------------

def is_available(name: str) -> bool:
    """Check whether *name* can be imported without side-effects."""
    obj = lazy_import(name)
    return not isinstance(obj, nullProxy)


def require(name: str, reason: str = ""):
    """
    Import *name* or raise immediately with a human-readable message.

    Use this in any manager __init__ where the dependency is non-optional.
    nullProxy is designed for graceful degradation — but a model manager
    that silently stores None and explodes later is not graceful, it's a trap.
    """
    obj = lazy_import(name)
    if isinstance(obj, nullProxy):
        hint = f" ({reason})" if reason else ""
        raise ImportError(
            f"Required package {name!r} is not installed{hint}. "
            f"Install it with:  pip install {name}"
        )
    return obj


# ---------------------------------------------------------------------------
# Torch  (with dynamo guard)
# ---------------------------------------------------------------------------

def get_torch():
    torch = lazy_import("torch")

    # Guard against PyTorch 2.x _refs dispatch recursion during
    # nn.Embedding init and similar paths.  This is a no-op if dynamo
    # is already disabled or on torch < 2.0.
    _dynamo = getattr(torch, "_dynamo", None)
    if _dynamo is not None and callable(getattr(_dynamo, "disable", None)):
        _dynamo.config.suppress_errors = True

    return torch


# ---------------------------------------------------------------------------
# spaCy
# ---------------------------------------------------------------------------

def get_spacy():
    return lazy_import("spacy")


# ---------------------------------------------------------------------------
# Whisper
# ---------------------------------------------------------------------------

def get_whisper():
    return lazy_import("openai_whisper")


# ---------------------------------------------------------------------------
# Transformers
# ---------------------------------------------------------------------------

def get_transformers(module=None):
    tf = lazy_import("transformers")
    if module is None:
        return tf

    _dispatch = {
        "AutoTokenizer": tf.AutoTokenizer,
        "AutoModelForSeq2SeqLM": tf.AutoModelForSeq2SeqLM,
        "AutoModelForCausalLM": tf.AutoModelForCausalLM,
        "GenerationConfig": tf.GenerationConfig,
        "BitsAndBytesConfig": tf.BitsAndBytesConfig,
        "pipeline": tf.pipeline,
        "LEDTokenizer": tf.LEDTokenizer,
        "LEDForConditionalGeneration": tf.LEDForConditionalGeneration,
        "T5TokenizerFast": tf.T5TokenizerFast,
        "T5ForConditionalGeneration": tf.T5ForConditionalGeneration,
    }
    if module not in _dispatch:
        raise KeyError(
            f"Unknown transformers sub-module {module!r}. "
            f"Available: {sorted(_dispatch)}"
        )
    return _dispatch[module]


# ---------------------------------------------------------------------------
# Sentence Transformers
# ---------------------------------------------------------------------------

def get_sentence_transformers(module=None):
    st = lazy_import("sentence_transformers")
    if module is None:
        return st

    _dispatch = {
        "SentenceTransformer": st.SentenceTransformer,
        "models": st.models,
        "cos_sim": lazy_import("sentence_transformers.util").cos_sim,
    }
    if module not in _dispatch:
        raise KeyError(
            f"Unknown sentence_transformers sub-module {module!r}. "
            f"Available: {sorted(_dispatch)}"
        )
    return _dispatch[module]


# ---------------------------------------------------------------------------
# KeyBERT
# ---------------------------------------------------------------------------

def get_keybert():
    """Return the KeyBERT *class*, not an instance.
    Downstream code that returns an initialised KeyBERT model must use
    a different name (e.g. get_keybert_instance) to avoid shadowing."""
    return lazy_import("keybert").KeyBERT


# ---------------------------------------------------------------------------
# MoviePy
# ---------------------------------------------------------------------------

def get_moviepy(module=None):
    mp = lazy_import("moviepy.editor")
    if module is None:
        return mp

    _dispatch = {
        "mp": mp,
        "VideoFileClip": mp.VideoFileClip,
    }
    if module not in _dispatch:
        raise KeyError(
            f"Unknown moviepy sub-module {module!r}. "
            f"Available: {sorted(_dispatch)}"
        )
    return _dispatch[module]

def get_tiktoken():
    return lazy_import("tiktoken")


# ---------------------------------------------------------------------------
# PyTesseract
# ---------------------------------------------------------------------------

def get_pytesseract():
    return lazy_import("pytesseract")


# ---------------------------------------------------------------------------
# PyPDF2
# ---------------------------------------------------------------------------

def get_pypdf2():
    return lazy_import("PyPDF2")


# ---------------------------------------------------------------------------
# EasyOCR
# ---------------------------------------------------------------------------

def get_easyocr():
    return lazy_import("easyocr")


# ---------------------------------------------------------------------------
# SpeechRecognition
# ---------------------------------------------------------------------------

def get_speech_recognition():
    return lazy_import("speech_recognition")


# ---------------------------------------------------------------------------
# PydubAudio
# ---------------------------------------------------------------------------

def get_pydub(module=None):
    pydub = lazy_import("pydub")
    if module is None:
        return pydub
    
    _dispatch = {
        "AudioSegment": lazy_import("pydub").AudioSegment,
        "silence": lazy_import("pydub.silence"),
    }
    if module not in _dispatch:
        raise KeyError(
            f"Unknown pydub sub-module {module!r}. "
            f"Available: {sorted(_dispatch)}"
        )
    return _dispatch[module]


# ---------------------------------------------------------------------------
# PaddleOCR
# ---------------------------------------------------------------------------

def get_paddleocr():
    return lazy_import("paddleocr").PaddleOCR


# ---------------------------------------------------------------------------
# pdf2image
# ---------------------------------------------------------------------------

def get_pdf2image(module=None):
    pdf2img = lazy_import("pdf2image")
    if module is None:
        return pdf2img
    
    _dispatch = {
        "convert_from_path": pdf2img.convert_from_path,
    }
    if module not in _dispatch:
        raise KeyError(
            f"Unknown pdf2image sub-module {module!r}. "
            f"Available: {sorted(_dispatch)}"
        )
    return _dispatch[module]
