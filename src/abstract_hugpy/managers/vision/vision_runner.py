# vision_runner.py
from .schemas import VisionRequest, VisionResult, VisionBackendConfig
from .vision_backends import build_backend


class VisionRunner:
    request_type = VisionRequest
    result_type = VisionResult

    def __init__(self, cfg):
        # cfg may be a ModelConfig (from dispatch) or a VisionBackendConfig directly.
        # Convert ModelConfig → VisionBackendConfig so build_backend always gets
        # the type it expects.
        if isinstance(cfg, VisionBackendConfig):
            backend_cfg = cfg
        else:
            backend_cfg = VisionBackendConfig(
                model_key=cfg.model_key,
                port=cfg.port,
                host=cfg.host or "http://127.0.0.1",
                timeout_s=float(cfg.timeout_s or 3600),
            )
        self.cfg = backend_cfg
        self.backend = build_backend(self.cfg)

    async def run(self, req: VisionRequest) -> VisionResult:
        if req.model_key != self.cfg.model_key:
            raise ValueError(
                f"VisionRunner bound to {self.cfg.model_key!r}, "
                f"got request for {req.model_key!r}"
            )
        return await self.backend.run(req)
