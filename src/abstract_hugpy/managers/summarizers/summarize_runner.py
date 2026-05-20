# runners/summarize_runner.py
class SummarizeRunner:
    request_type = SummarizeRequest
    result_type = SummarizeResult
    def __init__(self, model_key: str):
        self.model_key = model_key
    async def run(self, req):
        summary = await asyncio.to_thread(
            summarize, req.text, backend=self.model_key,
            preset=req.preset,
            max_length=req.max_length, min_length=req.min_length,
        )
        return SummarizeResult(
            request_id=req.request_id, model_key=req.model_key,
            summary=summary, chunks_processed=1,
        )

    async def stream(self, req, cancel_event):
        raise NotImplementedError("summarization is one-shot")
