#!/usr/bin/env python3
from .imports import *
import threading
_REPORT_LOCK = threading.Lock()

def record_downloaded_model(model: dict, dest: str = None, root: str = DEFAULT_ROOT,
                            report_path: str = None) -> dict:
    """Write one downloaded model into the discovery report so the registry —
    which derives from that report — shows it immediately, no full re-walk.

    Records the real on-disk dir (dest) so folder resolution is exact instead
    of a routed guess. Locked: monitor threads for concurrent downloads all
    read-modify-write the same file."""
    report_path = report_path or MODELS_DISCOVERY_PATH
    dest = dest or route_destination(model, root)
    hub_id = model.get("hub_id")
    name = model.get("name") or (hub_id.split("/")[-1] if hub_id else None)
    if not name:
        return {}

    primary = model.get("primary_task") or model.get("task")
    tasks = model.get("tasks") or ([primary] if primary else None)
    record = {
        "hub_id": hub_id,
        "framework": model.get("framework"),
        "primary_task": primary,
        "tasks": tasks,
        "filename": model.get("filename"),
        "include": model.get("include"),
        "model_max_length": model.get("model_max_length"),
        "dir": dest,
        "folder": os.path.relpath(dest, MODELS_HOME) if dest.startswith(MODELS_HOME) else dest,
    }
    with _REPORT_LOCK:
        report = safe_load_from_json(report_path) or {}
        report[name] = {**report.get(name, {}),
                        **{k: v for k, v in record.items() if v is not None}}
        safe_dump_to_file(data=report, file_path=report_path)
    return report[name]






def _clean_repo_id(hub_id: str) -> str:
    """Strip storage-path routing (family/task/...) back to owner/repo.

    cfg.hub_id sometimes carries the full storage path
    (gguf/text-generation/owner/repo) because discovery recorded the path as
    the id. HF wants just owner/repo. Drop leading family/task segments.
    """
    parts = hub_id.strip("/").split("/")
    FAMILIES = {"gguf", "transformers", "misc", "datasets", "models"}
    while len(parts) > 2 and parts[0] in FAMILIES:
        parts = parts[1:]                       # drop family
        if parts and parts[0] not in FAMILIES:
            parts = parts[1:]                   # drop the task that followed
    return "/".join(parts)

def _stamp(destination: str, key: str, model: dict[str, Any]) -> None:
    """Write hugpy.json into the destination so the model self-describes for
    discovery — identity no longer has to be inferred from the path later."""
    try:
        write_hugpy_marker(
            destination,
            hub_id=model.get("hub_id"),
            name=model.get("name") or key,
            framework=model.get("framework"),
            tasks=model.get("tasks"),
            primary_task=model.get("primary_task"),
            filename=model.get("filename"),
            include=model.get("include"),
            source="download",
        )
    except OSError as exc:
        print(f"  [warn] could not write hugpy.json: {exc}")


def download_one(model: dict[str, Any],root: str=None,model_key=None, dry_run: bool = False) -> None:
    
    hub_id = model.get("hub_id")
    framework = model.get("framework")
    filename = model.get("filename")
    include = model.get("include")
    if hub_id and not model_key:
        model_key = hub_id.split('/')[-1]
    root = root or DEFAULT_ROOT
    if not hub_id:
        print(f"[SKIP] {model_key}: missing hub_id")
        return

    destination = route_destination(model,root)
    repo_id, subfolder_from_hub = split_hub_id(hub_id)

    print()
    print(f"[MODEL] {model_key}")
    print(f"  framework: {framework}")
    print(f"  task:      {model.get('primary_task')}")
    print(f"  repo_id:   {repo_id}")
    if subfolder_from_hub:
        print(f"  subfolder: {subfolder_from_hub}")
    print(f"  dest:      {destination}")

    if dry_run:
        return

    os.makedirs(destination, exist_ok=True)

    # GGUF / single-file
    if framework == "llama_cpp":
        if filename:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                subfolder=subfolder_from_hub,
                local_dir=destination,
                local_dir_use_symlinks=False,
            )
            print(f"  downloaded file: {filename}")
            _stamp(destination, model_key, model)
            return

        if include:
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=include,
                local_dir=destination,
                local_dir_use_symlinks=False,
            )
            print(f"  downloaded pattern: {include}")
            _stamp(destination, model_key, model)
            return

        snapshot_download(
            repo_id=repo_id,
            local_dir=destination,
            local_dir_use_symlinks=False,
        )
        print("  downloaded full snapshot")
        _stamp(destination, model_key, model)
        return

    # Dataset
    if model.get("task") == "dataset" or model.get("primary_task") == "dataset":
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=destination,
            local_dir_use_symlinks=False,
        )
        print("  downloaded dataset snapshot")
        _stamp(destination, model_key, model)
        return

    # Transformers / full repo
    allow_patterns = include if include else None
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=allow_patterns,
        local_dir=destination,
        local_dir_use_symlinks=False,
    )
    if allow_patterns:
        print(f"  downloaded snapshot with pattern: {allow_patterns}")
    else:
        print("  downloaded full snapshot")
    _stamp(destination, model_key, model)


def download_dict_models() -> None:
    parser = argparse.ArgumentParser(description="Auto-distribute LLM downloads by framework/task.")
    parser.add_argument("manifest", type=str, help="Path to model manifest JSON.")
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT, help="LLM storage root.")
    parser.add_argument("--only", nargs="*", default=None, help="Optional model keys to download.")
    parser.add_argument("--dry-run", action="store_true", help="Print destinations without downloading.")
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", os.path.join(args.root, "cache", "huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(args.root, "cache", "huggingface", "hub"))
    os.environ.setdefault("TORCH_HOME", os.path.join(args.root, "cache", "torch"))
    os.environ.setdefault("PIP_CACHE_DIR", os.path.join(args.root, "cache", "pip"))

    # args.manifest is a str -> use open(), not args.manifest.open()
    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for key, model in manifest.items():
        if args.only and key not in args.only:
            continue
        try:
            download_one(key, model, args.root, dry_run=args.dry_run)
        except Exception as exc:
            print(f"[ERROR] {key}: {exc}")




def ensure_model(key: str, root: str = DEFAULT_ROOT) -> str:
    cfg = get_model_config(key)

    # Clean identity FIRST — both the repo HF pulls from and the path we
    # route to must use owner/repo, not the storage path.
    repo_id = _clean_repo_id(cfg.hub_id)

    # Route to framework/primary_task/hub_id using route_destination, the same
    # function download_one uses — single source of truth for layout. Build a
    # routing dict whose hub_id is the CLEAN id so we don't re-nest the prefix.
    routing = {
        "hub_id":       repo_id,
        "name":         getattr(cfg, "name", None) or key,
        "framework":    getattr(cfg, "framework", None),
        "primary_task": getattr(cfg, "primary_task", None) or "misc",
    }
    path = route_destination(routing,root)     # root/framework/task/owner/repo

    if model_looks_downloaded(path, cfg):
        return path

    os.makedirs(path, exist_ok=True)

    repo_id, subfolder = split_hub_id(repo_id)  # handle owner/repo/subfolder
    download_kwargs = {
        "repo_id": repo_id,
        "local_dir": path,
        "local_dir_use_symlinks": False,
    }
    if subfolder:
        download_kwargs["subfolder"] = subfolder
    if getattr(cfg, "include", None):
        download_kwargs["allow_patterns"] = cfg.include
    try:
        snapshot_download(**download_kwargs)

        write_hugpy_marker(
            path,
            hub_id=repo_id if not subfolder else f"{repo_id}/{subfolder}",
            name=getattr(cfg, "name", None) or key,
            framework=getattr(cfg, "framework", None),
            tasks=getattr(cfg, "tasks", None),
            primary_task=getattr(cfg, "primary_task", None),
            filename=getattr(cfg, "filename", None),
            include=getattr(cfg, "include", None),
            source="download",
        )
        return path
    except Exception as e:
        logger.info(f"Download Failed: {e}")
def ensure_models(models_dict_path=None):
    models_dict = get_models_dict(models_dict_path=models_dict_path)
    for model,values in models_dict.items():
        ensure_model(model)
