# /mnt/24T/hugging_face/new_hugs/keybertManager/functions/module/module_imports.py
# /mnt/24T/hugging_face/new_hugs/keybertManager/functions/module/module_imports.py

# lightweight stdlib imports stay
from ...imports import *
from collections import Counter
import os, re
import importlib
def lazy_import(name: str):
    """Load a module only when accessed."""
    return importlib.import_module(name)

# Torch
def get_torch():
    return lazy_import("torch")

# Transformers
def get_transformers():
    return lazy_import("transformers")

def get_pipeline():
    return get_transformers().pipeline

def get_AutoTokenizer():
    return get_transformers().AutoTokenizer

def get_AutoModelForSeq2SeqLM():
    return get_transformers().AutoModelForSeq2SeqLM

def get_LEDTokenizer():
    return get_transformers().LEDTokenizer

def get_LEDForConditionalGeneration():
    return get_transformers().LEDForConditionalGeneration

# Sentence Transformers
def get_sentence_transformers():
    return lazy_import("sentence_transformers")

def get_SentenceTransformer():
    return get_sentence_transformers().SentenceTransformer

def get_models():
    return get_sentence_transformers().models

def get_cos_sim():
    return get_sentence_transformers().util.cos_sim

# KeyBERT
def get_KeyBERT():
    return lazy_import("keybert").KeyBERT

