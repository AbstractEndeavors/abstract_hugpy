from .imports import *
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
