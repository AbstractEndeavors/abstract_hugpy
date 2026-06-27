"""Model config discovery: the hugpy.json marker is the source of truth for
identity; disk contents fill descriptive fields; registry is fallback.
"""

from .huggingface_api import *
from .call_api import *
from .imports import *
from huggingface_hub.errors import HFValidationError


# ---------------------------------------------------------------------------
# Clean hub_id resolution — one rule, used everywhere. Strips leading slashes
# and rejects ids that aren't owner/repo so they never reach the HF validator.
# ---------------------------------------------------------------------------
def clean_hub_id(directory: str, fallback: str = "") -> str:
    hub_id = hub_id_for(directory, fallback) or ""
    return hub_id.strip("/")


def is_valid_repo_id(hub_id: str) -> bool:
    hub_id = (hub_id or "").strip("/")
    return bool(hub_id) and "/" in hub_id

def base_present(base_model: str) -> bool:
    if not base_model:
        return True   # not an adapter; nothing to require
    base_dir = route_destination(
        {"hub_id": base_model, "framework": "transformers",
         "primary_task": "text-generation"}
    )
    return os.path.isdir(base_dir) and any(
        f.endswith(".safetensors") or f.endswith(".bin")
        for f in os.listdir(base_dir)
    )
# ---------------------------------------------------------------------------
# Registry lookup — exact / canonical match only
# ---------------------------------------------------------------------------
def get_config_from_folder(folder: str) -> Optional[dict]:
    target = _norm_folder(folder)
    if not target:
        return None
    for name, cfg in MODEL_REGISTRY.items():
        if normalize_folder(cfg.folder) == target:
            return cfg.to_dict()
    return None


# ---------------------------------------------------------------------------
# Main discovery walk
# ---------------------------------------------------------------------------
def get_all_configs(verbose: bool = False, get_code: bool = False,
                    get_files: bool = False, save_variables: bool = True
                    ) -> Dict[str, "ModelConfig"]:
    ALLCONFIGS: Dict[str, "ModelConfig"] = {}

    tasks_path = os.path.join(MODELS_HOME, "tasks.json")
    tasks_data = {}
    if os.path.isfile(tasks_path):
        tasks_data = safe_load_from_json(tasks_path) or {}

    model_dirs = exclude_dirs(get_model_dirs())
    shortnames = [d.replace(MODELS_HOME, "") for d in model_dirs]

    for directory in model_dirs:
        shortname = directory.replace(MODELS_HOME, "")
        folder = eatAll(shortname, "/")
        max_model_length = get_max_model_length(folder) or DEFAULT_MAX_TOKENS

        response_dir = get_response_dir(folder)
        _, resp_files = get_files_and_dirs(response_dir, allowed_exts=['.py'])
        if get_code and any(f.endswith('python_0.py') for f in resp_files):
            continue

        registry_cfg = get_config_from_folder(folder)
        marker = read_hugpy_marker(directory)

        guffs = get_guffs_in_dir(directory)
        configs = get_config_in_dir(directory)
        config_json = configs[0] if configs else None
        name = (marker or {}).get("name") or get_target_name(shortname, shortnames)

        hub_id = clean_hub_id(directory, folder)
        framework = (marker or {}).get("framework") or infer_framework(directory)
        tasks = (marker or {}).get("tasks") or tasks_data.get(name)
        primary_task = (marker or {}).get("primary_task") or (tasks[0] if tasks else None)

        discovered = {
            "name":             name,
            "hub_id":           hub_id,
            "folder":           folder,
            "framework":        framework,
            "filename":         (marker or {}).get("filename") or extract_gguf_filename(guffs, directory),
            "tasks":            tasks,
            "primary_task":     primary_task,
            "include":          (marker or {}).get("include"),
            "model_max_length": max_model_length,
            "port":             get_port(name),
            "host":             get_host(name),
        }

        merged, provenance = merge_disk_over_registry(discovered, registry_cfg)

        if merged.get("tasks") is None:
            merged["tasks"] = ["text-generation"]
            merged["primary_task"] = "text-generation"
            provenance["primary_task"] = "default"

        if verbose:
            print(f"[discover] {merged['name']}: {provenance}")

        name = merged["name"]
        if name and name not in ALLCONFIGS:
            ALLCONFIGS[name] = merged
            if get_code:
                call_and_code(model_config=merged, config_json=config_json, directory=directory)
        elif verbose:
            print(f"DUPLICATE NAME:\n{merged}")

        if get_files and name in ALLCONFIGS:
            ALLCONFIGS[name]['files'] = os.listdir(directory)

    if save_variables:
        if verbose:
            print(f"SAVING VARIABLES: {MODELS_DICT_PATH}")
        safe_dump_to_json(data=ALLCONFIGS, file_path=MODELS_DICT_PATH, indent=2)

    return ALLCONFIGS


# ---------------------------------------------------------------------------
# Resolvers — each independent; failure returns {} not an exception
# ---------------------------------------------------------------------------
def resolve_local_config(directory: str, hub_id: str) -> dict:
    path = os.path.join(directory, "config.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r") as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        "architectures":           cfg.get("architectures"),
        "model_type":              cfg.get("model_type"),
        "max_position_embeddings": cfg.get("max_position_embeddings"),
        "sliding_window":          cfg.get("sliding_window"),
        "rope_scaling":            cfg.get("rope_scaling"),
        "torch_dtype":             cfg.get("torch_dtype"),
        "vocab_size":              cfg.get("vocab_size"),
        # peft markers — present iff this is an adapter
        "peft_type":               cfg.get("peft_type"),
        "base_model":              cfg.get("base_model_name_or_path"),
    }


def resolve_local_tokenizer(directory: str, hub_id: str) -> dict:
    path = os.path.join(directory, "tokenizer_config.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r") as f:
            tk = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    raw = tk.get("model_max_length")
    if isinstance(raw, (int, float)) and 0 < raw < TOKENIZER_SENTINEL_THRESHOLD:
        return {"tokenizer_model_max_length": int(raw)}
    return {}


def resolve_hub_model_info(directory: str, hub_id: str, api: HfApi) -> dict:
    # Skip ids that aren't owner/repo — the HF validator raises on these,
    # and a malformed id can't resolve anything anyway.
    if not is_valid_repo_id(hub_id):
        return {}

    try:
        info = api.model_info(hub_id, files_metadata=False)
    except (HfHubHTTPError, HFValidationError) as exc:
        logger.warning("hub_model_info skipped for %r: %s", hub_id, exc)
        return {}

    params = getattr(info.safetensors, "total", None) if info.safetensors else None
    auto_model = getattr(info.transformers_info, "auto_model", None) if info.transformers_info else None

    card = info.card_data
    def _card(key):
        if card is None:
            return None
        return card.get(key) if isinstance(card, dict) else getattr(card, key, None)

    languages = _card("language")
    if isinstance(languages, str):
        languages = [languages]

    return {
        "pipeline_tag":     info.pipeline_tag,
        "library_name":     info.library_name,
        "auto_model_class": auto_model,
        "parameter_count":  params,
        "license":          _card("license"),
        "gated":            bool(info.gated) if info.gated is not None else None,
        "languages":        languages,
        "tags":             info.tags,
    }


# ---------------------------------------------------------------------------
# Resolver chain
# ---------------------------------------------------------------------------
ResolverFn = Callable[[str, str], dict]


def build_resolver_chain(*, api: Optional[HfApi] = None,
                         use_hub: bool = True) -> List[Tuple[str, ResolverFn]]:
    chain: List[Tuple[str, ResolverFn]] = [
        ("local_config",    resolve_local_config),
        ("local_tokenizer", resolve_local_tokenizer),
    ]
    if use_hub:
        hub_api = api or HfApi()
        chain.append(("hub_model_info", lambda d, h: resolve_hub_model_info(d, h, hub_api)))
    return chain


def enrich(directory: str, hub_id: str,
           chain: List[Tuple[str, ResolverFn]]) -> Tuple[ModelMetadata, Dict[str, str]]:
    valid_fields = {f.name for f in dataclass_fields(ModelMetadata)}
    merged: Dict[str, Any] = {"hub_id": hub_id}
    sources: Dict[str, str] = {"hub_id": "input"}

    for source_name, fn in chain:
        try:
            partial = fn(directory, hub_id)
        except Exception as exc:
            logger.warning("resolver %s failed for %r: %s", source_name, hub_id, exc)
            continue
        for field_name, value in partial.items():
            if value is None or field_name not in valid_fields:
                continue
            if merged.get(field_name) is None:
                merged[field_name] = value
                sources[field_name] = source_name

    return ModelMetadata(**merged), sources


def discover_model(save_json: bool = True, verbose: bool = True, use_hub: bool = True):
    discovered = {}
    chain = build_resolver_chain(use_hub=use_hub)

    for directory in get_model_dirs():
        file_parts = get_file_parts(directory)
        folder = directory[len(MODELS_HOME):]
        name = file_parts.get("basename")
        parent_dirname = file_parts.get("parent_dirname")

        hub_id = clean_hub_id(directory, directory[len(parent_dirname) + 1:])

        meta, meta_sources = enrich(directory, hub_id, chain)
        discovered[name] = {"name": name, "hub_id": hub_id, "folder": folder}
        discovered[name].update({
            "pipeline_tag":            meta.pipeline_tag,
            "library_name":            meta.library_name,
            "auto_model_class":        meta.auto_model_class,
            "architectures":           meta.architectures,
            "model_type":              meta.model_type,
            "max_position_embeddings": meta.max_position_embeddings,
            "model_max_length":        meta.tokenizer_model_max_length
                                       or meta.max_position_embeddings
                                       or DEFAULT_MAX_TOKENS,
            "parameter_count":         meta.parameter_count,
            "license":                 meta.license,
            "gated":                   meta.gated,
            "languages":               meta.languages,
        })
        if verbose:
            print(f"[enrich] {hub_id}: {meta_sources}")

    if save_json:
        safe_dump_to_file(data=discovered, file_path=MODELS_DICT_PATH)
    return discovered


def discover_models(save_json: bool = True, verbose: bool = True, use_hub: bool = True):
    """Walk the model tree and enrich each dir. Records the real on-disk
    location — discovery already walked it; don't throw it away."""
    discovered = {}
    chain = build_resolver_chain(use_hub=use_hub)

    for directory in get_model_dirs():
        file_parts = get_file_parts(directory)
        name = file_parts.get("basename")
        parent_dirname = file_parts.get("parent_dirname")

        hub_id = clean_hub_id(directory, directory[len(parent_dirname) + 1:])
        meta, meta_sources = enrich(directory, hub_id, chain)

        row = meta.to_dict()
        row["dir"] = directory                                  # absolute, ground truth
        try:
            row["folder"] = os.path.relpath(directory, MODELS_HOME)   # MODELS_HOME-relative
        except ValueError:
            row["folder"] = directory
        guffs = get_guffs_in_dir(directory)                     # capture gguf filename now
        if guffs:
            row["filename"] = extract_gguf_filename(guffs, directory)

        discovered[name] = row
        if verbose:
            print(f"[enrich] {hub_id}: {meta_sources}")

    if save_json:
        safe_dump_to_file(data=discovered, file_path=MODELS_DISCOVERY_PATH)
    return discovered
