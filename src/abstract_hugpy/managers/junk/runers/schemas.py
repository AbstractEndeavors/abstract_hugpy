# runners/schemas.py
class ChatRequest(TaskRequest):
    messages: list[Message]
    max_new_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False

class ChatResult(TaskResult):
    text: str
    finish_reason: str
    usage: dict | None = None

class SummarizeRequest(TaskRequest):
    text: str
    preset: str = "default"
    max_length: int | None = None
    min_length: int | None = None
    # ...

class SummarizeResult(TaskResult):
    summary: str
    chunks_processed: int

class TranscribeRequest(TaskRequest):
    audio_path: str | None = None
    audio_url: str | None = None
    language: str = "english"
    task: str | None = None  # "transcribe" or "translate"

class TranscribeResult(TaskResult):
    text: str
    segments: list[dict]
    language: str

class VisionRequest(TaskRequest):
    image_path: str
    prompt: str = "Analyze this image."
    max_new_tokens: int = 128

class VisionResult(TaskResult):
    text: str

class KeywordRequest(TaskRequest):
    text: str
    preset: str = "default"
    top_n: int = 10
    # ...

class KeywordResult(TaskResult):
    primary: list[str]
    secondary: list[str]
    density: dict[str, float]
