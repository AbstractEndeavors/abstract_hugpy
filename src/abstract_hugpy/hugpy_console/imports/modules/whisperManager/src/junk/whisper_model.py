import os
import whisper
import moviepy.editor as mp
from moviepy.editor import VideoFileClip
from .config import DEFAULT_PATHS
from abstract_utilities import get_logFile,SingletonMeta
logger = get_logFile(__name__)
DEFAULT_WHISPER_MODEL_PATH = "/mnt/24T/hugging_face/modules/whisper_base"
class whisperManager(metaclass=SingletonMeta):
    def __init__(self,module_size: str = "base", whisper_model_path: str = None):
        if not hasattr(self, 'initialized') or module_size != self.module_size:
            self.whisper_model_path = whisper_model_path or DEFAULT_WHISPER_MODEL_PATH
            self.module_size = module_size
            self.whisper_model = whisper.load_model(self.module_size, download_root=self.whisper_model_path)
def get_whisper_model(module_size: str = "base", whisper_model_path: str = None):
    whisper_mgr = whisperManager(module_size=module_size,whisper_model_path=whisper_model_path)
    return whisper_mgr.whisper_model


def whisper_transcribe(
    audio_path: str,
    model_size: str = "small",
    language: str = "english",
    use_silence: bool = True,
    task=None,
    whisper_model_path: str = None
):
    model = get_whisper_model(module_size=model_size, whisper_model_path=whisper_model_path)
    return model.transcribe(audio_path, language=language)


