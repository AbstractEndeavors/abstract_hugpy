import os
import os.path as osp
import math
from dataclasses import dataclass, field
from typing import Optional, List

from PIL import Image

from .imports import (
    get_torch,
    get_transformers,
    get_logFile,
    require,
    DEFAULT_PATHS,
)

logger = get_logFile("vision_coder")


# Qwen2.5-VL: 14x14 ViT patches + 2x2 spatial merge -> each visual token = 28x28 px
QWEN_PATCH = 28
QWEN_PIXELS_PER_TOKEN = QWEN_PATCH * QWEN_PATCH  # 784


# ---- bad-input sentinels (defense at the boundary) -------------------------

_BAD_PATH_STRINGS = frozenset({
    "", "[object object]", "undefined", "null", "none",
})


def _coerce_image_path(value) -> str:
    if not isinstance(value, str):
        raise TypeError(
            f"image_path must be a string, got {type(value).__name__}: {value!r}"
        )
    cleaned = value.strip()
    if cleaned.lower() in _BAD_PATH_STRINGS:
        raise ValueError(
            f"image_path looks like a serialization artifact, not a real path: {value!r}"
        )
    if not osp.exists(cleaned):
        raise FileNotFoundError(f"Image not found: {cleaned}")
    return cleaned


# ---- config ----------------------------------------------------------------

@dataclass(frozen=True)
class VisionCoderConfig:
    model_dir: str
    device: str
    torch_dtype: object              # torch.dtype, but no torch import at module top
    min_tokens: int = 64
    max_tokens: int = 384
    local_files_only: bool = True

    @property
    def min_pixels(self) -> int:
        return self.min_tokens * QWEN_PIXELS_PER_TOKEN

    @property
    def max_pixels(self) -> int:
        return self.max_tokens * QWEN_PIXELS_PER_TOKEN


# ---- pure helpers ----------------------------------------------------------

def resolve_qwen_vl_path(module_path: Optional[str] = None) -> str:
    """Resolve the model directory. Pure: returns a string, raises on miss."""
    candidates = []
    if module_path:
        candidates.append(str(module_path))

    from_env = DEFAULT_PATHS.get("qwen_vl")
    if from_env:
        candidates.append(str(from_env))

    candidates.append("/var/www/hugging_face/modules/Qwen/Qwen2.5-VL-7B-Instruct")

    for c in candidates:
        if osp.exists(c):
            return c

    raise FileNotFoundError(
        "Could not resolve local Qwen2.5-VL model path. "
        "Set MODEL_QWEN_VL=/var/www/hugging_face/modules/Qwen/Qwen2.5-VL-7B-Instruct"
    )


def _pick_device_and_dtype(torch, device: Optional[str], dtype) -> tuple[str, object]:
    """Choose device and dtype safely. Never fp16 on CPU (no native kernel path)."""
    chosen_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is not None:
        return chosen_device, dtype

    if chosen_device == "cuda":
        return chosen_device, torch.float16
    if hasattr(torch.cpu, "is_bf16_supported") and torch.cpu.is_bf16_supported():
        return chosen_device, torch.bfloat16
    return chosen_device, torch.float32


def fit_to_token_budget(image: Image.Image, max_tokens: int) -> Image.Image:
    """Downscale (never upscale) so visual-token count <= max_tokens.
    Snaps to multiples of QWEN_PATCH so the processor doesn't smart-resize again."""
    w, h = image.size
    pixel_budget = max_tokens * QWEN_PIXELS_PER_TOKEN
    if w * h <= pixel_budget:
        return image
    scale = math.sqrt(pixel_budget / (w * h))
    nw = max(QWEN_PATCH, int(w * scale) // QWEN_PATCH * QWEN_PATCH)
    nh = max(QWEN_PATCH, int(h * scale) // QWEN_PATCH * QWEN_PATCH)
    return image.resize((nw, nh), Image.LANCZOS)


def build_config(
    model_dir: Optional[str] = None,
    device: Optional[str] = None,
    torch_dtype=None,
    min_tokens: int = 64,
    max_tokens: int = 384,
) -> VisionCoderConfig:
    torch = require("torch", reason="VisionCoder requires PyTorch")
    chosen_device, chosen_dtype = _pick_device_and_dtype(torch, device, torch_dtype)
    resolved = resolve_qwen_vl_path(model_dir)
    return VisionCoderConfig(
        model_dir=resolved,
        device=chosen_device,
        torch_dtype=chosen_dtype,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )


# ---- the loaded model object -----------------------------------------------

class VisionCoder:
    """Holds a loaded Qwen2.5-VL model + processor. Constructed once at boot."""

    def __init__(self, cfg: VisionCoderConfig):
        require("transformers", reason="VisionCoder requires HuggingFace transformers")

        self.cfg = cfg
        logger.info(
            "VisionCoder loading model=%s device=%s dtype=%s token_budget=[%d,%d]",
            cfg.model_dir, cfg.device, cfg.torch_dtype, cfg.min_tokens, cfg.max_tokens,
        )

        Qwen2_5_VLForConditionalGeneration = get_transformers(
            "Qwen2_5_VLForConditionalGeneration"
        )
        AutoProcessor = get_transformers("AutoProcessor")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg.model_dir,
            torch_dtype=cfg.torch_dtype,
            trust_remote_code=True,
            local_files_only=cfg.local_files_only,
        ).to(cfg.device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            cfg.model_dir,
            trust_remote_code=True,
            local_files_only=cfg.local_files_only,
            min_pixels=cfg.min_pixels,
            max_pixels=cfg.max_pixels,
        )

    def analyze_image(
        self,
        image_path: str,
        prompt: str = "Analyze this image.",
        max_new_tokens: int = 128,
        max_tokens: Optional[int] = None,
    ) -> str:
        torch = get_torch()

        path = _coerce_image_path(image_path)
        image = Image.open(path).convert("RGB")
        budget = max_tokens if max_tokens is not None else self.cfg.max_tokens
        image = fit_to_token_budget(image, budget)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt", padding=True,
        )
        inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,            # deterministic; flip on for sampled outputs
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated = output_ids[:, prompt_len:]
        return self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]

    def analyze_images(
        self,
        image_paths: List[str],
        prompt: str = "Analyze this image.",
        max_new_tokens: int = 128,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        """Sequential today; here so callers can pass a list without surprise."""
        return [
            self.analyze_image(p, prompt=prompt, max_new_tokens=max_new_tokens, max_tokens=max_tokens)
            for p in image_paths
        ]


# ---- module-level registry (single instance per process) -------------------

_INSTANCE: Optional[VisionCoder] = None


def get_vision_coder(
    module_path: Optional[str] = None,
    torch_dtype=None,
    max_tokens: int = 384,
    min_tokens: int = 64,
) -> VisionCoder:
    """Lazy boot. First call builds; later calls return the same instance.
    If you need a different config, build a VisionCoder directly and hold the ref."""
    global _INSTANCE
    if _INSTANCE is None:
        cfg = build_config(
            model_dir=module_path,
            torch_dtype=torch_dtype,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        )
        _INSTANCE = VisionCoder(cfg)
    return _INSTANCE


# ---- legacy entry point ----------------------------------------------------

def deepcoder_image_analysis(
    image_path,
    prompt: str = "please describe this image",
    max_new_tokens: int = 100,
    module_path: Optional[str] = None,
    torch_dtype=None,
    max_tokens: Optional[int] = None,
):
    vision = get_vision_coder(module_path=module_path, torch_dtype=torch_dtype)
    return vision.analyze_image(
        image_path=image_path,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        max_tokens=max_tokens,
    )
