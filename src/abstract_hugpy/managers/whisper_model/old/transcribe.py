from __future__ import annotations

import json
import os
import subprocess

from abstract_utilities import derive_media_type

from .imports import (
    get_whisper,
    get_logFile,
    SingletonMeta,
    DEFAULT_PATHS,
)
from .media_artifacts import create_media_workspace

logger = get_logFile(__name__)

DEFAULT_WHISPER_MODEL_PATH = DEFAULT_PATHS["whisper-large-v3"]


class whisperManager(metaclass=SingletonMeta):
    def __init__(
        self,
        module_size: str = "base",
        whisper_model_path: str | None = None,
    ):
        current_module_size = getattr(self, "module_size", None)
        current_model_path = getattr(self, "whisper_model_path", None)

        next_model_path = whisper_model_path or DEFAULT_WHISPER_MODEL_PATH

        should_load = (
            not getattr(self, "initialized", False)
            or current_module_size != module_size
            or current_model_path != next_model_path
        )

        if should_load:
            self.whisper_model_path = next_model_path
            self.module_size = module_size
            self.whisper_model = get_whisper().load_model(
                self.module_size,
                download_root=self.whisper_model_path,
            )
            self.initialized = True


def get_whisper_model(
    module_size: str = "base",
    whisper_model_path: str | None = None,
):
    whisper_mgr = whisperManager(
        module_size=module_size,
        whisper_model_path=whisper_model_path,
    )
    return whisper_mgr.whisper_model


def extract_audio_from_video_ffmpeg(
    video_path: str,
    audio_path: str,
) -> str:
    if not os.path.isfile(video_path):
        raise ValueError(f"Video file does not exist: {video_path}")

    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        audio_path,
    ]

    logger.info(f"Extracting audio from {video_path} to {audio_path}")

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed to extract audio.\n\n"
            f"Command:\n{' '.join(command)}\n\n"
            f"stderr:\n{result.stderr}"
        )

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file was not created: {audio_path}")

    return audio_path


def whisper_transcribe(
    audio_path: str,
    model_size: str = "small",
    language: str | None = "english",
    task: str = "transcribe",
    whisper_model_path: str | None = None,
):
    if not os.path.isfile(audio_path):
        raise ValueError(f"Audio file does not exist: {audio_path}")

    if task not in {"transcribe", "translate"}:
        raise ValueError(f"Unsupported Whisper task: {task}")

    model = get_whisper_model(
        module_size=model_size,
        whisper_model_path=whisper_model_path,
    )

    options = {"task": task}

    if language:
        options["language"] = language

    return model.transcribe(audio_path, **options)


def save_transcript_outputs(
    whisper_result: dict,
    transcript_json_path: str,
    transcript_text_path: str,
) -> None:
    with open(transcript_json_path, "w", encoding="utf-8") as file:
        json.dump(whisper_result, file, indent=2, ensure_ascii=False)

    text = whisper_result.get("text", "")
    with open(transcript_text_path, "w", encoding="utf-8") as file:
        file.write(text.strip() + "\n")


def transcribe_file_with_workspace(
    file_path: str,
    model_size: str = "small",
    language: str | None = "english",
    task: str = "transcribe",
    whisper_model_path: str | None = None,
    output_root: str | None = None,
    copy_source: bool = False,
):
    if not os.path.isfile(file_path):
        raise ValueError(f"Media file does not exist: {file_path}")

    workspace = create_media_workspace(
        source_path=file_path,
        output_root=output_root,
        copy_source=copy_source,
    )

    media_type = derive_media_type(file_path)

    if media_type == "audio":
        audio_path = file_path
        workspace.manifest.set_file("audio", audio_path)

    elif media_type == "video":
        audio_path = str(workspace.audio_path)
        extract_audio_from_video_ffmpeg(
            video_path=file_path,
            audio_path=audio_path,
        )
        workspace.manifest.set_file("audio", audio_path)

    else:
        raise ValueError(f"Unsupported media type for transcription: {media_type}")

    whisper_result = whisper_transcribe(
        audio_path=audio_path,
        model_size=model_size,
        language=language,
        task=task,
        whisper_model_path=whisper_model_path,
    )

    save_transcript_outputs(
        whisper_result=whisper_result,
        transcript_json_path=str(workspace.transcript_json_path),
        transcript_text_path=str(workspace.transcript_text_path),
    )

    workspace.manifest.set_file("transcript_json", workspace.transcript_json_path)
    workspace.manifest.set_file("transcript_text", workspace.transcript_text_path)
    workspace.manifest.set_metadata("model_size", model_size)
    workspace.manifest.set_metadata("language", language)
    workspace.manifest.set_metadata("task", task)
    workspace.save_manifest()

    return {
        "workspace_dir": str(workspace.root_dir),
        "audio_path": audio_path,
        "transcript_json_path": str(workspace.transcript_json_path),
        "transcript_text_path": str(workspace.transcript_text_path),
        "manifest_path": str(workspace.root_dir / "manifest.json"),
        "whisper_result": whisper_result,
    }
