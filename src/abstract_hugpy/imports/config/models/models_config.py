"""models_config.py — the registry.

MODELS is the authoritative base: the real, curated models. Discovery finds
everything else on disk (test downloads) and appends it at build time. Staples
are never overwritten — a discovered row is skipped if its model_key OR its
cleaned hub_id already belongs to a staple (prevents same-path collisions like
Falconsai-text-summarization vs a discovered text_summarization).

Build order:
    MODELS (curated)  +  discovery report (derived)  ->  ModelConfig registry

Import is cheap: it merges MODELS with whatever discovery report already exists
on disk. To re-walk the model tree (HF metadata, network), call
refresh_registry() explicitly — e.g. on hugpy module startup.
"""

from .imports import *

logger = get_logFile(__name__)


# ===========================================================================
# Base registry — the real models. Authoritative.
# ===========================================================================
MODELS = {
    "text_summarization": {
        "model_max_length": 512, "include": None, "name": "Falconsai-text-summarization",
        "framework": "transformers", "hub_id": "Falconsai/text_summarization", "filename": None,
        "folder": "Falconsai/text_summarization", "tasks": ["text-summarization"],
        "primary_task": "text-summarization", "port": None,
    },
    "led-large-16384": {
        "model_max_length": 16384, "include": None, "name": "led-large-16384",
        "framework": "transformers", "hub_id": "allenai/led-large-16384", "filename": None,
        "folder": "allenai/led-large-16384", "tasks": ["text-summarization"],
        "primary_task": "text-summarization", "port": None,
    },
    "flan-t5-xl": {
        "model_max_length": 1024, "include": None, "name": "flan-t5-xl",
        "framework": "transformers", "hub_id": "google/flan-t5-xl", "filename": None,
        "folder": "google/flan-t5-xl", "tasks": ["text-summarization", "text2text-generation"],
        "primary_task": "text-summarization", "port": None,
    },
    "all-minilm-l6-v2": {
        "model_max_length": 512, "include": None, "name": "all-minilm-l6-v2",
        "framework": "transformers", "hub_id": "sentence-transformers/all-minilm-l6-v2",
        "filename": None, "folder": "sentence-transformers/all-minilm-l6-v2",
        "tasks": ["feature-extraction", "sentence-similarity"],
        "primary_task": "feature-extraction", "port": None,
    },
    "gte-large-en-v1.5": {
        "model_max_length": 8192, "include": None, "name": "gte-large-en-v1.5",
        "framework": "transformers", "hub_id": "Alibaba-NLP/gte-large-en-v1.5", "filename": None,
        "folder": "Alibaba-NLP/gte-large-en-v1.5",
        "tasks": ["feature-extraction", "sentence-similarity"],
        "primary_task": "feature-extraction", "port": None,
    },
    "whisper-large-v3": {
        "model_max_length": 448, "include": None, "name": "whisper-large-v3",
        "framework": "transformers", "hub_id": "openai/whisper-large-v3", "filename": None,
        "folder": "openai/whisper-large-v3", "tasks": ["automatic-speech-recognition"],
        "primary_task": "automatic-speech-recognition", "port": None,
    },
}


# ===========================================================================
# Derivation — discovery/manifest row -> ModelConfig-ready dict.
# Pure; no torch, no runner-stack import (so building the registry never drags
### the inference stack in). RUNNER_PAIRS mirrors FRAMEWORK_RUNNERS statically.
# ===========================================================================
DEFAULT_MAX_TOKENS_LOCAL = DEFAULT_MAX_TOKENS

_FAMILIES = {"gguf", "transformers", "misc", "datasets", "models"}


def _clean_repo_id(hub_id):
    """Strip storage-path leakage (gguf/text-generation/owner/repo, leading
    slashes) back to owner/repo — the only shape HF and routing accept."""
    parts = (hub_id or "").strip("/").split("/")
    while len(parts) > 2 and parts[0] in _FAMILIES:
        parts = parts[1:]
        if parts and parts[0] not in _FAMILIES:
            parts = parts[1:]
    return "/".join(parts)

def base_present(base_model: str) -> bool:
    """True if a PEFT adapter's base model is actually on disk.

    Non-adapters (base_model falsy) pass trivially. An adapter passes only
    if route_destination's base dir exists AND holds real weights — a bare
    dir with just a config doesn't count.
    """
    if not base_model:
        return True
    base_dir = route_destination(
        {"hub_id": base_model,
         "framework": "transformers",
         "primary_task": "text-generation"}
    )
    if not os.path.isdir(base_dir):
        return False
    try:
        return any(
            f.endswith(".safetensors") or f.endswith(".bin")
            for f in os.listdir(base_dir)
        )
    except OSError:
        return False


_SEQ2SEQ = {"t5", "led", "bart", "pegasus", "mbart", "mt5", "longt5"}
_EMBED   = {"bert", "new", "roberta", "mpnet", "nomic_bert"}
_ASR     = {"whisper"}
_VISION  = {"qwen2_5_vl", "minicpmv4_6", "mllama", "idefics3", "internvl"}

def _safe_path_part(value):
    value = value.strip().replace("\\", "/")
    value = re.sub(r"[^A-Za-z0-9._/\-]+", "_", value)
    value = re.sub(r"/+", "/", value)
    return value.strip("/")

def _runtime_folder(framework, hub_id, include=None, filename=None):
    framework = (framework or "").lower().strip()
    if framework == "llama_cpp": return "gguf"
    if filename and filename.lower().endswith(".gguf"): return "gguf"
    if include:
        pats = include if isinstance(include, list) else [include]
        if any("gguf" in p.lower() for p in pats): return "gguf"
    return "transformers" if framework == "transformers" else "misc"

def _routed_folder(framework, task, hub_id, filename=None, include=None):
    """Predicted MODELS_HOME-relative folder — only used when the model isn't
    on disk yet, so there's no real dir to record."""
    if task == "dataset": return None
    return f"{_runtime_folder(framework, hub_id, include, filename)}/{_safe_path_part(task)}/{_safe_path_part(hub_id)}"

def _resolve_folder(row, framework, task, hub_id, filename, include):
    """Real dir wins; then an already-routed folder; then a prediction."""
    abs_dir = row.get("dir")
    if abs_dir and MODELS_HOME:
        rel = os.path.relpath(abs_dir, MODELS_HOME)
        if not rel.startswith(".."):
            return rel
    f = row.get("folder")
    if f and len(f.strip("/").split("/")) >= 3:   # looks like runtime/task/owner/repo already
        return f.strip("/")
    return _routed_folder(framework, task, hub_id, filename, include) or hub_id
def _derive_framework(name, hub_id, row):
    if row.get("framework"):
        return row["framework"]
    blob = f"{name} {hub_id}".lower()
    tags = [t.lower() for t in (row.get("tags") or [])]
    return "llama_cpp" if ("gguf" in blob or "gguf" in tags) else "transformers"


def _derive_tasks(framework, row):
    tasks = row.get("tasks")
    if tasks:
        return tasks if isinstance(tasks, list) else [tasks]
    if framework == "llama_cpp":
        return ["text-generation"]                      # all the gguf runner serves
    pt = row.get("pipeline_tag") or row.get("primary_task") or row.get("task")
    if pt in HF_TASK_TO_TASKS:
        return list(HF_TASK_TO_TASKS[pt])
    m = (row.get("model_type") or "").lower()
    if m in _SEQ2SEQ: return ["text-summarization", "text2text-generation"]
    if m in _EMBED:   return ["feature-extraction", "sentence-similarity"]
    if m in _ASR:     return ["automatic-speech-recognition"]
    if m in _VISION:  return ["image-text-to-text", "text-generation"]
    return ["text-generation"]                          # conservative floor


def derive_model_config_row(name, row):
    """One discovery/manifest row -> ModelConfig-ready dict, or (None, reason)."""
    hub_id = _clean_repo_id(row.get("hub_id") or row.get("folder") or name)
    if not hub_id or "/" not in hub_id:
        return None, f"unusable hub_id {row.get('hub_id')!r}"

    # PEFT adapter gate: an adapter is a delta on a base model. It needs
    # base_model_name_or_path, and that base must be on disk to serve.
    # base-less or base-absent adapters are dropped here so they never
    # enter the registry and detonate inside from_pretrained on first use.
    peft_base = row.get("base_model")
    if peft_base and not base_present(peft_base):
        return None, f"peft adapter base {peft_base!r} not on disk; acquire it first"

    framework = _derive_framework(name, hub_id, row)
    tasks = _derive_tasks(framework, row)
    primary = row.get("primary_task") if row.get("primary_task") in tasks else tasks[0]
    bad = [t for t in tasks if (framework, t) not in RUNNER_PAIRS]
    if bad:
        return None, f"({framework},{bad}) has no runner"
    folder = _resolve_folder(row, framework, primary, hub_id,
                             row.get("filename"), row.get("include"))
    return {
        "name": row.get("name") or name, "model_key": name,
        "hub_id": hub_id, "folder": folder,
        "dir": row.get("dir"),
        "framework": framework, "tasks": tasks, "primary_task": primary,
        "base_model": peft_base,                 # None for ordinary models
        "model_max_length": row.get("model_max_length")
            or row.get("tokenizer_model_max_length")
            or row.get("max_position_embeddings") or DEFAULT_MAX_TOKENS_LOCAL,
        "filename": row.get("filename"), "include": row.get("include"),
        "port": row.get("port"), "host": row.get("host"),
    }, None

def _absorb_disk(staple, disc):
    """Disk facts from a discovered row override a staple's hand-written guesses."""
    for k in ("dir", "folder", "filename"):
        if disc.get(k):
            staple[k] = disc[k]

def merge_discovery_into_models(discovery, base=None):
    base = base if base is not None else MODELS
    merged = {k: dict(v) for k, v in base.items()}
    hub_to_key = {_clean_repo_id(v.get("hub_id")): k for k, v in base.items()}
    dropped = []
    for name, row in (discovery or {}).items():
        hub = _clean_repo_id(row.get("hub_id") or row.get("folder") or name)
        if name in merged:                       # same key as a staple
            _absorb_disk(merged[name], row); continue
        if hub in hub_to_key:                     # same hub_id as a staple
            _absorb_disk(merged[hub_to_key[hub]], row)
            dropped.append((name, f"merged into staple {hub_to_key[hub]} (same hub_id)"))
            continue
        cfg, why = derive_model_config_row(name, row)
        if cfg is None:
            dropped.append((name, why)); continue
        merged[name] = row if "dir" in row else dict(row)
        merged[name].setdefault("model_key", name)
        hub_to_key[hub] = name
    # derive every merged row (staples now carry absorbed disk facts)
    out, drops2 = {}, []
    for name, row in merged.items():
        cfg, why = derive_model_config_row(name, row)
        (out.__setitem__(name, cfg) if cfg else drops2.append((name, why)))
    return out, dropped + drops2


# ===========================================================================
# ModelConfig assembly — identical validation path as before.
# ===========================================================================
def assess_config(cls, values):
    """Build cls if values can form a valid instance, else False. Never raises."""
    flds = {f.name: f for f in fields(cls)}
    for f in flds.values():
        required = f.default is MISSING and f.default_factory is MISSING
        if required and values.get(f.name) in (None, "", []):
            return False
    out = {}
    for name, f in flds.items():
        if name in values:
            out[name] = values[name]
        elif f.default is not MISSING:
            out[name] = f.default
        else:
            out[name] = f.default_factory()
    return cls(**out)


def get_model_values(config, dict_return=False, return_dict=False):
    if dict_return or return_dict:
        return config.to_dict()
    return config


def get_assessed_model_config(values, dict_return=False, return_dict=False):
    assessed = assess_config(ModelConfig, values)
    if assessed is False:
        return False
    return get_model_values(assessed, dict_return=dict_return, return_dict=return_dict)


def update_model_config_dict(model_key=None, values=None, dict_obj=None,
                             dict_return=False, return_dict=False, key=None):
    dict_obj = dict_obj if dict_obj is not None else {}
    model_key = model_key or key
    values = dict(values or {})
    values["model_key"] = model_key
    config = get_assessed_model_config(values, dict_return=dict_return, return_dict=return_dict)
    if config is False:
        logger.warning("registry: %s failed ModelConfig assessment, skipped", model_key)
        return dict_obj
    dict_obj[model_key] = config
    return dict_obj


def _load_discovery_report(path=None):
    """Read the on-disk discovery report. Prefer the descriptive report; fall
    back to the registry-shaped manifest. Either shape works."""
    for candidate in (path, MODELS_DISCOVERY_PATH, MODELS_DICT_PATH):
        if candidate and os.path.isfile(candidate):
            data = safe_load_from_json(candidate)
            if data:
                return data
    return {}


def get_models_dict(models_dict_path=None, dict_return=False, return_dict=False,
                    discovery=None):
    """Build the registry: MODELS + discovery (test downloads).

    discovery=None -> read the report on disk. Pass a dict to merge an
    in-memory discovery result (e.g. straight from a fresh walk)."""
    dict_return = dict_return or return_dict
    report = discovery if discovery is not None else _load_discovery_report(models_dict_path)
    merged, dropped = merge_discovery_into_models(report)

    for model_key, why in dropped:
        logger.info("registry: dropped %s (%s)", model_key, why)

    nudict = {}
    for model_key, values in merged.items():
        nudict = update_model_config_dict(
            model_key=model_key, values=values, dict_obj=nudict, dict_return=dict_return
        )
    return nudict


# ===========================================================================
# Registry — built at import from MODELS + existing discovery report.
# ===========================================================================
MODEL_REGISTRY: Dict[str, ModelConfig] = get_models_dict()
MODEL_REGISTRY_DICT: Dict[str, dict] = get_models_dict(dict_return=True)


def get_model_registry(dict_return=False, return_dict=False):
    dict_return = dict_return or return_dict
    return MODEL_REGISTRY_DICT if dict_return else MODEL_REGISTRY


def refresh_registry(run_discovery=True):
    """Re-walk the model tree and rebuild MODEL_REGISTRY in place. Call this on
    hugpy startup. run_discovery=False just re-reads the existing report.

    Late import of discover_models avoids a circular import at module load."""
    global MODEL_REGISTRY, MODEL_REGISTRY_DICT
    report = None
    if run_discovery:
        try:
            from ...apis.get_module import discover_models
            report = discover_models(save_json=True, verbose=False, use_hub=True)
        except Exception as exc:
            logger.warning("refresh_registry: discovery walk failed (%s); "
                           "falling back to on-disk report", exc)
    MODEL_REGISTRY = get_models_dict(discovery=report)
    MODEL_REGISTRY_DICT = get_models_dict(dict_return=True, discovery=report)
    return MODEL_REGISTRY
