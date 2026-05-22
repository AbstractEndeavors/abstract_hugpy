from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def slugify(value: str, fallback: str = "media") -> str:
    value = value.strip()
    value = re.sub(r"[^\w.\- ]+", "_", value)
    value = re.sub(r"\s+", "_", value)
    value = value.strip("._-")
    return value or fallback


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    parent = path.parent
    stem = path.stem
    suffix = path.suffix

    for index in range(1, 10_000):
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not create unique path for: {path}")


@dataclass
class MediaArtifactManifest:
    source_path: str
    workspace_dir: str
    created_at: str
    files: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def set_file(self, key: str, path: str | Path | None) -> None:
        if path is not None:
            self.files[key] = str(path)

    def set_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def save(self) -> str:
        manifest_path = Path(self.workspace_dir) / "manifest.json"
        manifest_path.write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return str(manifest_path)


@dataclass
class MediaWorkspace:
    source_path: Path
    root_dir: Path
    manifest: MediaArtifactManifest

    @property
    def audio_path(self) -> Path:
        return self.root_dir / "audio.wav"

    @property
    def transcript_json_path(self) -> Path:
        return self.root_dir / "transcript.json"

    @property
    def transcript_text_path(self) -> Path:
        return self.root_dir / "transcript.txt"

    @property
    def frames_dir(self) -> Path:
        path = self.root_dir / "frames"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def frame_context_path(self) -> Path:
        return self.root_dir / "frame_context.json"

    def save_manifest(self) -> str:
        return self.manifest.save()


def create_media_workspace(
    source_path: str,
    output_root: str | None = None,
    copy_source: bool = False,
    overwrite: bool = False,
) -> MediaWorkspace:
    source = Path(source_path).expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(f"Source media file not found: {source}")

    parent = Path(output_root).expanduser().resolve() if output_root else source.parent
    workspace_name = f"{slugify(source.stem)}.assets"
    workspace_dir = parent / workspace_name

    if workspace_dir.exists() and not overwrite:
        workspace_dir = unique_path(workspace_dir)

    workspace_dir.mkdir(parents=True, exist_ok=True)

    manifest = MediaArtifactManifest(
        source_path=str(source),
        workspace_dir=str(workspace_dir),
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    if copy_source:
        copied_source = workspace_dir / source.name
        shutil.copy2(source, copied_source)
        manifest.set_file("source_copy", copied_source)

    manifest.set_file("source", source)

    workspace = MediaWorkspace(
        source_path=source,
        root_dir=workspace_dir,
        manifest=manifest,
    )

    workspace.save_manifest()
    return workspace
