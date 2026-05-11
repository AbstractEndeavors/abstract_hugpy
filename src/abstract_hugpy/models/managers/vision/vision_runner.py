from .vision_coder import get_vision_coder
from .schemas import VisionRequest,VisionResult
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
