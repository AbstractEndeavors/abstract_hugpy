import os
DEFAULT_REL_FILE_PATH = "datas/modules.json"
class ModuleSource:
    def __init__(self, modules: dict=None):
        self.modules_path = None
        self.absdir = get_caller_dir()
        if not modules:
            self.modules_path = os.path.join(self.absdir,DEFAULT_REL_FILE_PATH)
        if modules and os.path.isfile(modules):
            self.modules_path = modules
        if self.modules_path and os.path.isfile(self.modules_path):
            modules = safe_read_from_json(self.modules_path)
        self.modules = modules

    def resolve(self, name: str):
        """
        Returns (source, cache_dir, kind)
        - source   -> string to pass to from_pretrained (local path if exists; else repo_id)
        - cache_dir-> where to cache; default to configured local path if present
        - kind     -> "model" | "dataset" | "other" (lets you special-case non-models)
        """
        if name not in self.modules:
            raise KeyError(f"Unknown module '{name}'")

        rec = self.modules[name]
        path = rec.get("path")
        repo_id = rec.get("repo_id")  # must be a string like "org/model"
        handle = rec.get("handle", name)

        # Decide kind (very light heuristic)
        kind = "dataset" if "/dataset" in (repo_id or "").lower() or "dataset" in handle.lower() else "model"
        if name == "huggingface":
            kind = "other"

        # Prefer local path if it exists; else use repo id
        if path and os.path.isdir(path):
            source = path
            cache_dir = path  # keep snapshots alongside
        else:
            source = repo_id
            cache_dir = path if path else None  # optional; OK to be None

        return source, cache_dir, kind
