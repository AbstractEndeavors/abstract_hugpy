from .constants import *
from .imports import *
# ---------------------------------------------------------------------------
# Model Paths — One row of everything we know. All Optional — partial fills are valid.
# ---------------------------------------------------------------------------


def join_path(*paths):
    return os.path.join(*paths)

def get_model_home(models_home=None):
    return models_home or MODELS_HOME



def write_hugpy_marker(directory, *, hub_id, name=None, framework=None,
                       tasks=None, primary_task=None, filename=None,
                       include=None, source="download", **extra):
    """Stamp a model dir with its identity. Single source of truth for what
    this model IS — discovery keys on it instead of guessing from the path."""
    if tasks is not None and not isinstance(tasks, list):
        tasks = [tasks]
    payload = {
        "hub_id": hub_id,
        "name": name or (hub_id.split("/")[-1] if hub_id else None),
        "framework": framework,
        "tasks": tasks,
        "primary_task": primary_task or (tasks[0] if tasks else None),
        "filename": filename,
        "include": include,
        "source": source,                       # "download" | "custom"
        "stamped_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, HUGPY_MARKER)
    safe_dump_to_json(file_path=path,data=payload, indent=2)
    return path


def read_hugpy_marker(directory):
    """Return the declared identity dict, or None if unstamped/unreadable."""
    path = os.path.join(directory, HUGPY_MARKER)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def has_hugpy_marker(directory):
    return os.path.isfile(os.path.join(directory, HUGPY_MARKER))


def hub_id_for(directory, fallback=None):
    """Repo id from the declared marker; explicit sources, path slice last.

    1. hugpy.json (authoritative — declared at download/custom time)
    2. legacy .llm_storage_installed.json marker
    3. config.json _name_or_path
    4. fallback (path slice) — only if nothing self-describes
    """
    marker = read_hugpy_marker(directory)
    if marker and marker.get("hub_id"):
        return marker["hub_id"]

    legacy = os.path.join(directory, ".llm_storage_installed.json")
    if os.path.isfile(legacy):
        try:
            with open(legacy, "r", encoding="utf-8") as f:
                hid = json.load(f).get("hub_id")
            if hid:
                return hid
        except (OSError, json.JSONDecodeError):
            pass

    cfg = os.path.join(directory, "config.json")
    if os.path.isfile(cfg):
        try:
            with open(cfg, "r", encoding="utf-8") as f:
                nop = json.load(f).get("_name_or_path")
            if nop and "/" in nop and not os.path.isabs(nop):
                return nop
        except (OSError, json.JSONDecodeError):
            pass

    return fallback


def backfill_markers(get_model_dirs, hub_id_fallback=lambda d: None, verbose=True):
    """One-time: stamp a hugpy.json into any model dir that lacks one, using
    whatever identity can be salvaged (legacy marker, config.json, fallback).
    After this, every dir is self-describing."""
    stamped, skipped = [], []
    for directory in get_model_dirs():
        if has_hugpy_marker(directory):
            skipped.append(directory)
            continue
        hub_id = hub_id_for(directory, hub_id_fallback(directory))
        if not hub_id:
            if verbose:
                print(f"[backfill] no hub_id resolvable, skipping: {directory}")
            continue
        framework = None
        cfg = read_hugpy_marker(directory)  # None here, but keep shape
        write_hugpy_marker(directory, hub_id=hub_id, source="backfill")
        stamped.append(directory)
        if verbose:
            print(f"[backfill] stamped {hub_id} -> {directory}")
    return {"stamped": stamped, "skipped": skipped}
