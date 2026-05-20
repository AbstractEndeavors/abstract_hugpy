from .keybert_model import *
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
