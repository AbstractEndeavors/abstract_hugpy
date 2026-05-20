from pathlib import Path
from constants import *

def resolve_hf_model_dir(base_dir: str | Path) -> Path:
    base = Path(base_dir)

    if (base / "config.json").exists():
        return base

    snapshots = base / "snapshots"
    if snapshots.exists():
        candidates = [
            p for p in snapshots.iterdir()
            if p.is_dir() and (p / "config.json").exists()
        ]

        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime)

    raise FileNotFoundError(f"No usable Hugging Face model dir found under: {base}")
for model,cfg in MODEL_REGISTRY.items():
    input(cfg.to_dict())
    base_dir = get_model_path(cfg.folder)
    hf_model_dir = resolve_hf_model_dir(base_dir)
    input(hf_model_dir)
