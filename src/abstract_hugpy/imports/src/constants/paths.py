from .imports import *
from .constants import *
from .hugpy_marker import *
def safe_path_part(value: str) -> str:
    value = value.strip().replace("\\", "/")
    value = re.sub(r"[^A-Za-z0-9._/\-]+", "_", value)
    value = re.sub(r"/+", "/", value)
    return value.strip("/")
def safe_name(value: str) -> str:
    value = value.strip()
    value = value.replace("\\", "/")
    value = re.sub(r"[^A-Za-z0-9._/\-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_/")
def split_hub_id(hub_id: str) -> tuple[str, str | None]:
    parts = hub_id.strip("/").split("/")
    if len(parts) <= 2:
        return hub_id, None
    repo_id = "/".join(parts[:2])
    subfolder = "/".join(parts[2:])
    return repo_id, subfolder
def runtime_folder(framework: str, hub_id: str, include: Any = None, filename: str | None = None) -> str:
    framework = framework.lower().strip()

    if framework == "llama_cpp":
        return "gguf"

    if filename and filename.lower().endswith(".gguf"):
        return "gguf"

    if include:
        patterns = include if isinstance(include, list) else [include]
        if any("gguf" in pattern.lower() for pattern in patterns):
            return "gguf"

    if framework == "transformers":
        return "transformers"

    return "misc"

# ---------------------------------------------------------------------------
# Model Paths — One row of everything we know. All Optional — partial fills are valid.
# ---------------------------------------------------------------------------
def is_model_dir(directory: str) -> bool:
    """A model dir declares itself with a hugpy.json. Fall back to weight
    markers for dirs not yet stamped (legacy / first-run before backfill)."""
    if os.path.isfile(os.path.join(directory, HUGPY_MARKER)):
        return True
    try:
        entries = os.listdir(directory)
    except (OSError, NotADirectoryError):
        return False
    return any(e == "config.json" or e.lower().endswith((".gguf", ".safetensors"))
               for e in entries)

def get_model_dirs(models_home=None):
    root = get_model_home(models_home=models_home)
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if not is_directory_excluded(join_path(dirpath, d))]
        if is_model_dir(dirpath):
            found.append(dirpath)
            dirnames[:] = []     # model dir is a leaf; don't descend
    return found

def hub_id_for(directory: str, fallback: str | None = None) -> str | None:
    """Repo id from the declared marker; path slice only if unstamped."""
    marker = read_hugpy_marker(directory)
    if marker and marker.get("hub_id"):
        return marker["hub_id"]
    return fallback


def get_model_dirs(models_home=None):
    """Walk MODELS_HOME and return every directory that actually holds a model,
    regardless of how deep it sits. Skips excluded dirs and never lists files."""
    root = get_model_home(models_home=models_home)
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        # prune excluded subtrees in-place so os.walk doesn't descend them
        dirnames[:] = [d for d in dirnames
                       if not is_directory_excluded(join_path(dirpath, d))]
        if is_model_dir(dirpath):
            found.append(dirpath)
            dirnames[:] = []          # don't descend into a model's own files
    return found

def get_hub_id_from_directory(directory, models_home=None):
    """Repo id = the path of the model dir relative to its task dir.
    (directory is the longer path; strip the task-dir prefix off the FRONT.)"""
    root = get_model_home(models_home=models_home)
    rel = directory[len(root):].strip("/")
    parts = rel.split("/")
    # rel = family/task/owner/repo[/subfolder]; drop family+task, keep the rest
    return "/".join(parts[2:]) if len(parts) > 2 else rel

def is_directory_excluded(directory):
    base = os.path.basename(directory.rstrip("/"))
    if base in EXCLUDE_DIR_NAMES:
        return True
    return any(base.startswith(p) for p in EXCLUDE_DIR_PREFIXES)

def exclude_dirs(directories):
    return [d for d in directories if not is_directory_excluded(d)]
def resolve_task(model: dict) -> str:
    """Effective task across all dict shapes:
       ModelConfig.to_dict() -> primary_task (+ tasks list)
       /repos/download dict   -> primary_task
       legacy manifest        -> task (singular)
    First non-empty wins; misc is the floor."""
    pt = model.get("primary_task")
    if pt:
        return pt
    tasks = model.get("tasks")
    if tasks:
        return tasks[0] if isinstance(tasks, list) else tasks
    return model.get("task") or "misc"
def route_destination(model: dict, root: str = DEFAULT_ROOT) -> str:
    hub_id    = model.get("hub_id") or model.get("name") or model.get("folder") or ""
    framework = (model.get("framework") or "").strip()
    task      = resolve_task(model)
    hub_path  = safe_path_part(hub_id)

    if task == "dataset":
        return os.path.join(root, "datasets", hub_path)

    runtime = runtime_folder(framework, hub_id,
                             include=model.get("include"),
                             filename=model.get("filename"))
    return os.path.join(root, "models", runtime, safe_path_part(task), hub_path)
