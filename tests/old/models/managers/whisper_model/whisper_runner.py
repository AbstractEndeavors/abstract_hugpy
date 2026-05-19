
# runners/transcribe_runner.py
class WhisperRunner:
    request_type = TranscribeRequest
    result_type = TranscribeResult

    def __init__(self, model_key: str):
        self.model_key = model_key

    async def run(self, req):
        if req.audio_url:
            from .stream_whisper_result import whisper_transcribe_url_stream
            result = await asyncio.to_thread(
                whisper_transcribe_url_stream,
                url=req.audio_url, language=req.language, task=req.task,
            )
        elif req.audio_path:
            result = await asyncio.to_thread(
                whisper_transcribe,
                audio_path=req.audio_path, language=req.language, task=req.task,
            )
        else:
            raise ValueError("must provide audio_path or audio_url")

        return TranscribeResult(
            request_id=req.request_id, model_key=req.model_key,
            text=result["text"], segments=result.get("segments", []),
            language=result.get("language", req.language),
        )
