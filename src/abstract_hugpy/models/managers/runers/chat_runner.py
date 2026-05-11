# runners/chat_runner.py
class DeepCoderChatRunner:
    request_type = ChatRequest
    result_type = ChatResult

    def __init__(self, model_key: str):
        self.model_key = model_key
        cfg = build_deepcoder_runtime(model_key=model_key)
        self.coder = REGISTRY.get(cfg)

    async def run(self, req: ChatRequest) -> ChatResult:
        messages = [m.model_dump() for m in req.messages]
        try:
            text = await asyncio.to_thread(
                self.coder.generate_text,
                messages,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                do_sample=req.do_sample,
                use_chat_template=True,
            )
            return ChatResult(
                request_id=req.request_id, model_key=req.model_key,
                text=text, finish_reason="stop",
            )
        except Exception as exc:
            return ChatResult(
                request_id=req.request_id, model_key=req.model_key,
                ok=False, error=f"{type(exc).__name__}: {exc}",
                text="", finish_reason="error",
            )

    async def stream(self, req, cancel_event):
        async for event in self.coder.stream_chat(req, cancel_event=cancel_event):
            yield event


class LlamaCppChatRunner:
    request_type = ChatRequest
    result_type = ChatResult

    def __init__(self, model_key: str):
        self.model_key = model_key
        self.runner = get_llama_runner(model_key)  # your existing singleton

    async def run(self, req):
        messages = [m.model_dump() for m in req.messages]
        text = await asyncio.to_thread(
            self.runner.generate_text, messages,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature, top_p=req.top_p,
            do_sample=req.do_sample, use_chat_template=True,
        )
        return ChatResult(
            request_id=req.request_id, model_key=req.model_key,
            text=text, finish_reason="stop",
        )

    async def stream(self, req, cancel_event):
        async for event in self.runner.stream_chat(req, cancel_event=cancel_event):
            yield event


# runners/summarize_runner.py
class SummarizeRunner:
    request_type = SummarizeRequest
    result_type = SummarizeResult

    def __init__(self, model_key: str):
        # Map your model_keys → existing backend keys
        self.backend_key = {
            "summarizer": "t5",
            "flan": "flan",
            "bigbird": "falconsai",  # or wire bigbird as its own backend
        }[model_key]
        self.model_key = model_key

    async def run(self, req):
        summary = await asyncio.to_thread(
            summarize, req.text, backend=self.backend_key,
            preset=req.preset,
            max_length=req.max_length, min_length=req.min_length,
        )
        return SummarizeResult(
            request_id=req.request_id, model_key=req.model_key,
            summary=summary, chunks_processed=1,
        )

    async def stream(self, req, cancel_event):
        raise NotImplementedError("summarization is one-shot")


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


# runners/vision_runner.py — wraps VisionCoder
class VisionRunner:
    request_type = VisionRequest
    result_type = VisionResult

    def __init__(self, model_key: str):
        self.model_key = model_key
        self.vision = get_vision_coder()

    async def run(self, req):
        text = await asyncio.to_thread(
            self.vision.analyze_image,
            image_path=req.image_path, prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
        )
        return VisionResult(
            request_id=req.request_id, model_key=req.model_key, text=text,
        )


# runners/keyword_runner.py
class KeywordRunner:
    request_type = KeywordRequest
    result_type = KeywordResult

    def __init__(self, model_key: str):
        self.model_key = model_key

    async def run(self, req):
        refined = await asyncio.to_thread(
            refine_keywords, req.text, preset=req.preset, top_n=req.top_n,
        )
        return KeywordResult(
            request_id=req.request_id, model_key=req.model_key,
            primary=refined.primary, secondary=refined.secondary,
            density=refined.density,
        )
