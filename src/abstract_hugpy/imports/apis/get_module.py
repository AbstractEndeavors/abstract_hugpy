"""Model config discovery: filesystem is the source of truth.

Discovery walks the modules directory, classifies each model folder by what's
actually on disk, then merges the registry on top *as a fallback* for fields
disk can't infer (task, hub_id when it differs from folder, include globs).

Resolution order, highest precedence first:
    1. Whatever's actually on disk (folder name, framework markers, gguf filename)
    2. Hand-coded MODEL_REGISTRY entry (task, hub_id, include glob)
    3. Conservative defaults

`_provenance` is attached to each merged dict at discovery time so logs can
answer "where did this field come from" without rerunning. It's stripped
before ModelConfig(**...) construction.
"""
import os
from typing import Optional, Dict, List, Tuple
from .huggingface_api import *
from .call_api import *
from .imports import *

def is_directory_excluded(directory):
    for part in EXCLUDE_DIR_NAMES:
        if part in directory:
            return True
    for part in EXCLUDE_DIR_PREFIXES:
        if part in directory:
            return True
        
def exclude_dirs(directories):
    return [directory for directory in directories if not is_directory_excluded(directory)]


# ---------------------------------------------------------------------------
# Path / name helpers
# ---------------------------------------------------------------------------




def _norm_folder(folder: str) -> str:
    """Canonical form for folder comparison: stripped, no leading/trailing slashes."""
    return eatAll(folder or "", "/").strip()

def get_name(targetname: str, shortnames: List[str]) -> List[str]:
    shortnames = [s for s in shortnames if s != targetname]
    targetparts = get_parts(targetname)
    for shortname in shortnames:
        shortparts = get_parts(shortname)
        targetparts = get_unique_part(targetparts, shortparts)
    return targetparts

def get_target_name(shortname: str, shortnames: List[str]) -> str:
    targetname = get_name(shortname, shortnames)
    targetnames = targetname or get_parts(shortname)
    return targetnames[-1]


# ---------------------------------------------------------------------------
# Registry lookup — exact / canonical match only (FIX #1)
# ---------------------------------------------------------------------------

def get_config_from_folder(folder: str) -> Optional[dict]:
    """Find a registry entry whose folder matches exactly (after normalization).

    The previous version used startswith() in BOTH directions, which is too
    permissive — 'flux' would match 'flux_dev', 'Qwen2.5-Coder-3B' would
    match 'Qwen2.5-Coder-3B-GGUF', etc. Once discovery ran on a moving
    filesystem you'd silently stamp the wrong config onto a new folder.

    Exact-match-only here. If you legitimately need aliases later, add an
    explicit `aliases: list[str]` field to ModelConfig — don't go back to
    fuzzy matching.
    """
    target = _norm_folder(folder)
    if not target:
        return None

    for name, cfg in MODEL_REGISTRY.items():
        cfg_folder = _norm_folder(cfg.folder)
        if cfg_folder == target:
            return cfg.to_dict()
    return None


# ---------------------------------------------------------------------------
# Filesystem inspection
# ---------------------------------------------------------------------------

def get_guffs_in_dir(directory: str) -> List[str]:
    return [
        os.path.join(directory, item)
        for item in os.listdir(directory)
        if item.endswith(".gguf") or item.endswith(".GGUF")
    ]


def get_config_in_dir(directory: str) -> List[str]:
    return [
        os.path.join(directory, item)
        for item in os.listdir(directory)
        if item.endswith("config.json")
    ]


def infer_framework(directory: str) -> Optional[str]:
    """Decide a framework from what's actually on disk.

    Returns None when ambiguous so the registry can fill in. The previous
    rule was 'llama_cpp if filename else None', which mislabels anything
    that isn't a GGUF — including diffusion models like FLUX that happened
    to be tagged llama_cpp by hand.
    """
    try:
        files = os.listdir(directory)
    except OSError:
        return None

    has_ext = lambda ext: any(f.endswith(ext) for f in files)

    if has_ext(".gguf") or has_ext(".GGUF"):
        return "llama_cpp"
    if has_ext(".safetensors") or has_ext(".bin"):
        return "transformers"
    if has_ext(".onnx"):
        return "onnx"
    return None


def extract_gguf_filename(guffs: List[str], directory: str) -> Optional[str]:
    """Return the GGUF path relative to its folder.

    The previous version did `directory.replace(guff, "")` — backwards,
    since guff is the full joined path and directory doesn't contain it.
    That returned the directory unchanged and then ate slashes off it.

    For multi-shard models (Qwen3-Coder-Next splits into 4 files), this
    returns the path including the shard subdirectory, matching what's
    already in your registry: 'Qwen3-Coder-Next-Q4_K_M/...-00001-of-00004.gguf'.
    """
    if not guffs:
        return None
    # Pick the first shard if there are multiples; sort so it's stable.
    chosen = sorted(guffs)[0]
    return os.path.relpath(chosen, start=directory)


# ---------------------------------------------------------------------------
# Merge — disk wins, registry fills gaps (FIX #2)
# ---------------------------------------------------------------------------

# Fields where a discovered (non-None) value should override the registry.
# Keeps task/hub_id/include in registry territory because those can't be
# inferred from a directory listing.
_DISK_AUTHORITATIVE = ("name", "folder", "framework", "filename")


def _merge_disk_over_registry(
    discovered: dict,
    registry: Optional[dict],
) -> Tuple[dict, Dict[str, str]]:
    """Merge with disk taking precedence over registry on _DISK_AUTHORITATIVE.

    Returns (merged_dict, provenance_map) where provenance_map[field] is
    'disk' | 'registry' | 'default'. Provenance is for logging only — not
    fed into ModelConfig.
    """
    registry = registry or {}
    merged: dict = {}
    prov: Dict[str, str] = {}

    all_fields = set(registry) | set(discovered)
    for field in all_fields:
        disk_val = discovered.get(field)
        reg_val = registry.get(field)

        if field in _DISK_AUTHORITATIVE and disk_val is not None:
            merged[field] = disk_val
            prov[field] = "disk"
        elif reg_val is not None:
            merged[field] = reg_val
            prov[field] = "registry"
        elif disk_val is not None:
            merged[field] = disk_val
            prov[field] = "disk"
        else:
            merged[field] = None
            prov[field] = "default"

    return merged, prov
def get_port(name):
    key = f"{name}_PORT"
    return get_env_value(key)
    
# ---------------------------------------------------------------------------
# Main discovery walk
# ---------------------------------------------------------------------------

def get_all_configs(verbose: bool = False,get_code: bool =False,get_files: bool =False,save_variables=True) -> Dict[str, "ModelConfig"]:
    """Walk modules_dir, discover models from disk, merge registry as fallback.

    Set verbose=True to print provenance per field — useful when reconciling
    a hand-coded registry against new on-disk reality.
    """
    ALLCONFIGS: Dict[str, "ModelConfig"] = {}


    dirs, files = get_files_and_dirs(
        MODELS_HOME, allowed_exts=[".json", ".GGUF", ".gguf"]
    )

    # Folders that contain at least one config.json or gguf
    directories = list({
        os.path.dirname(f)
        for f in files
        if f.endswith("config.json")
        or f.endswith(".gguf")
        or f.endswith(".GGUF")
    })
    directories = exclude_dirs(directories)
    shortnames = [d.replace(MODELS_HOME, "") for d in dirs]
    for directory in directories:
        shortname = directory.replace(MODELS_HOME, "")
        folder = eatAll(shortname, "/")
        max_model_length = get_max_model_length(folder) or DEFAULT_MAX_TOKENS
        response_dir = get_response_dir(folder)
        dirs,files = get_files_and_dirs(response_dir,allowed_exts=['.py'])
        if get_code:
            if [file for file in files if file.endswith('python_0.py')]:
                continue
        # --- look up registry entry by exact folder match ------------------
        registry_cfg = get_config_from_folder(folder)

        # --- inspect disk --------------------------------------------------
        guffs = get_guffs_in_dir(directory)
        configs = get_config_in_dir(directory)
        config_json=None
        disk_config_path = configs[0] if configs else None
        name = get_target_name(shortname, shortnames)
        discovered = {
            "name":      name,
            "hub_id":    folder,
            "folder":    folder,
            "framework": infer_framework(directory),
            "filename":  extract_gguf_filename(guffs, directory),
            "task":      None,    # disk can't tell us; let registry fill
            "include":   None,    # same — registry-only field
            "model_max_length": max_model_length,
            "port": get_port(name)
        }

        merged, provenance = _merge_disk_over_registry(discovered, registry_cfg)
        
        # --- conservative defaults for anything still None -----------------
        if merged.get("task") is None:
            merged["task"] = "code-generation"
            provenance["task"] = "default"
        if configs:
            config_json = configs[0]
        if verbose:
            print(f"[discover] {merged['name']}: {provenance}")
            
        name = merged["name"]
        print(name)
        if name and name not in ALLCONFIGS:
            ALLCONFIGS[name] = merged
            if get_code:
                call_and_code(model_config=merged, config_json=config_json,directory=directory)
            
        else:
            if verbose:     
                print(f"DUPLICATE NAME:\n{merged}")
        if get_files:
            print(f"FILES:")
            files = os.listdir(directory)
            ALLCONFIGS[name]['files'] = files
            for item in files:
                print(item)
    if save_variables:
        safe_dump_to_json(data=ALLCONFIGS,file_path=MODELS_DICT_PATH,indent=2)        
    return ALLCONFIGS


def get_max_model_length(folder):
    all_max_values = []

    module_dir = os.path.join(MODELS_HOME,folder)
    dirlist = os.listdir(module_dir)
    files = [os.path.join(module_dir,file) for file in dirlist if file.endswith('.json')]
    for file in files:
        data = safe_load_from_json(file)
        values = list(find_keys_by_type(data, int, path=()))
        max_values = [value for value in values if 'max_' in value[-1]] or []
        for max_value in max_values:
            if max_value[-1] == "model_max_length":
                return get_any_value(data,"model_max_length")
        all_max_values+=max_values
    return DEFAULT_MAX_TOKENS


           






