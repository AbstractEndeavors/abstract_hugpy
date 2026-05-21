## Whisper transcription utilities (src/utilities/transcribe.py)
from .imports import (
    os,
    get_moviepy,
    get_whisper,
    get_logFile,
    SingletonMeta,
    DEFAULT_PATHS,
)
from abstract_utilities import derive_media_type

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

    options = {
        "task": task,
    }

    if language:
        options["language"] = language

    return model.transcribe(audio_path, **options)


def extract_audio_from_video(
    video_path: str,
    audio_path: str | None = None,
) -> str:
    """Extract audio from a video file using moviepy."""
    if not os.path.isfile(video_path):
        raise ValueError(f"Video file does not exist: {video_path}")

    if audio_path is None:
        video_directory = os.path.dirname(video_path) or "."
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = os.path.join(video_directory, f"{base_name}.wav")

    if os.path.isdir(audio_path):
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = os.path.join(audio_path, f"{base_name}.wav")

    try:
        logger.info(f"Extracting audio from {video_path} to {audio_path}")

        video = get_moviepy("mp").VideoFileClip(video_path)

        if video.audio is None:
            raise ValueError(f"Video has no audio track: {video_path}")

        video.audio.write_audiofile(audio_path)
        video.close()

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file was not created: {audio_path}")

        logger.info(f"Audio extracted successfully: {audio_path}")
        return audio_path

    except Exception:
        logger.exception(f"Error extracting audio from {video_path}")
        raise


def transcribe_from_video(
    video_path: str,
    audio_path: str | None = None,
    model_size: str = "small",
    language: str | None = "english",
    task: str = "transcribe",
    whisper_model_path: str | None = None,
):
    extracted_audio_path = extract_audio_from_video(
        video_path=video_path,
        audio_path=audio_path,
    )

    return whisper_transcribe(
        audio_path=extracted_audio_path,
        model_size=model_size,
        language=language,
        task=task,
        whisper_model_path=whisper_model_path,
    )


def transcribe_file(
    file_path: str,
    model_size: str = "small",
    language: str | None = "english",
    task: str = "transcribe",
    whisper_model_path: str | None = None,
    audio_path: str | None = None,
):
    if not os.path.isfile(file_path):
        raise ValueError(f"Media file does not exist: {file_path}")

    media_type = derive_media_type(file_path)

    if media_type == "audio":
        return whisper_transcribe(
            audio_path=file_path,
            model_size=model_size,
            language=language,
            task=task,
            whisper_model_path=whisper_model_path,
        )

    if media_type == "video":
        return transcribe_from_video(
            video_path=file_path,
            audio_path=audio_path,
            model_size=model_size,
            language=language,
            task=task,
            whisper_model_path=whisper_model_path,
        )

    raise ValueError(f"Unsupported media type for transcription: {media_type}")
