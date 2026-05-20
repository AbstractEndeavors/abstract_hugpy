# vision_runner.py

import asyncio
from .vision_coder import get_vision_coder
from .schemas import VisionRequest, VisionResult


class VisionRunner:
    request_type = VisionRequest
    result_type = VisionResult

    def __init__(self, model_key: str):
        self.model_key = model_key
        # passes the key through so the right model is loaded/cached
        self.vision = get_vision_coder(model_key=model_key)

    async def run(self, req: VisionRequest) -> VisionResult:
        text = await asyncio.to_thread(
            self.vision.analyze_image,
            image_path=req.image_path,
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            max_tokens=req.max_tokens,
        )
        return VisionResult(
            request_id=req.request_id,
            model_key=req.model_key,
            text=text,
        )
