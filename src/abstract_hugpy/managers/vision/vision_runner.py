# vision_runner.py
# vision_runner.py
from .schemas import VisionRequest, VisionResult, VisionBackendConfig
from .vision_backends import build_backend


class VisionRunner:
    request_type = VisionRequest
    result_type = VisionResult

    def __init__(self, cfg: VisionBackendConfig):
        self.cfg = cfg
        self.backend = build_backend(self.cfg)

    async def run(self, req: VisionRequest) -> VisionResult:
        if req.model_key != self.cfg.model_key:
            raise ValueError(
                f"VisionRunner bound to {self.cfg.model_key!r}, "
                f"got request for {req.model_key!r}"
            )
        return await self.backend.run(req)
