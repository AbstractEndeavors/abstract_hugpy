import importlib
import sys
from types import ModuleType
from functools import lru_cache


# --------------------------------------------------------------------------
# Core lazy import utility
# --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def lazy_import(name: str) -> ModuleType:
    """
    Lazily import a module only once per process.
    Returns cached module from sys.modules if already imported.
    """
    if name in sys.modules:
        return sys.modules[name]
    return importlib.import_module(name)


# --------------------------------------------------------------------------
# Torch
# --------------------------------------------------------------------------
def get_torch():
    return lazy_import("torch")

def get_whisper():
    return lazy_import("whisper")


# --------------------------------------------------------------------------
# Transformers
# --------------------------------------------------------------------------
def get_transformers():
    return lazy_import("transformers")

def get_pipeline():
    return getattr(get_transformers(), "pipeline")

def get_T5TokenizerFast():
    return getattr(get_transformers(), "T5TokenizerFast")

def get_T5ForConditionalGeneration():
    return getattr(get_transformers(), "T5ForConditionalGeneration")

def get_GenerationConfig():
    return getattr(get_transformers(), "GenerationConfig")

def get_AutoTokenizer():
    return getattr(get_transformers(), "AutoTokenizer")

def get_AutoModelForSeq2SeqLM():
    return getattr(get_transformers(), "AutoModelForSeq2SeqLM")

def get_LEDTokenizer():
    return getattr(get_transformers(), "LEDTokenizer")

def get_LEDForConditionalGeneration():
    return getattr(get_transformers(), "LEDForConditionalGeneration")

def get_AutoModelForCausalLM():
    return getattr(get_sentence_transformers(), "AutoModelForCausalLM")


# --------------------------------------------------------------------------
# Sentence Transformers
# --------------------------------------------------------------------------
def get_sentence_transformers():
    return lazy_import("sentence_transformers")

def get_SentenceTransformer():
    return getattr(get_sentence_transformers(), "SentenceTransformer")

def get_models():
    return getattr(get_sentence_transformers(), "models")

def get_cos_sim():
    return getattr(get_sentence_transformers().util, "cos_sim")


# --------------------------------------------------------------------------
# KeyBERT
# --------------------------------------------------------------------------
def get_KeyBERT():
    return getattr(lazy_import("keybert"), "KeyBERT")


# --------------------------------------------------------------------------
# DeepCoder / Transformer Internals
# --------------------------------------------------------------------------
def get_modeling_outputs():
    return lazy_import("transformers.modeling_outputs")

def get_CausalLMOutputWithPast():
    outputs = get_modeling_outputs()
    return getattr(outputs, "CausalLMOutputWithPast")
