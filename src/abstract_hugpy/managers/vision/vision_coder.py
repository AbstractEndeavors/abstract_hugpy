# vision_coder.py

import math
from dataclasses import dataclass
from typing import Optional, List

from PIL import Image
from .schemas import *
from .imports import (
    get_torch,
    get_transformers,
    get_logFile,
    require,
    DEFAULT_PATHS,
    VISION_MODELS_REGISTRY,
    DEFAULT_VISION_MODEL,
    get_model_path
)

logger = get_logFile("vision_coder")

QWEN_PATCH = 28
QWEN_PIXELS_PER_TOKEN = QWEN_PATCH * QWEN_PATCH  # 784

_BAD_PATH_STRINGS = frozenset({
    "", "[object object]", "undefined", "null", "none",
})

import io, base64
from PIL import Image
# vision_coder.py



def open_image_from_request(req: "VisionRequest") -> Image.Image:
    if req.image_path is not None:
        return Image.open(req.image_path).convert("RGB")
    if req.image_b64 is not None:
        raw = base64.b64decode(req.image_b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    raise ValueError(
        f"VisionRequest {req.request_id!r} has neither image_path nor image_b64; "
        "schema validation should have caught this earlier"
    )
def _open_image(req: VisionRequest) -> Image.Image:
    if req.image_path is not None:
        return Image.open(req.image_path).convert("RGB")
    return Image.open(io.BytesIO(base64.b64decode(req.image_b64))).convert("RGB")
def _coerce_image_path(value) -> str:
    import os.path as osp
    if not isinstance(value, str):
        raise TypeError(f"image_path must be a string, got {type(value).__name__}: {value!r}")
    cleaned = value.strip()
    if cleaned.lower() in _BAD_PATH_STRINGS:
        raise ValueError(f"image_path looks like a serialization artifact: {value!r}")
    if not osp.exists(cleaned):
        raise FileNotFoundError(f"Image not found: {cleaned}")
    return cleaned


@dataclass(frozen=True)
class VisionCoderConfig:
    model_key: str                   # registry key, e.g. "Qwen2.5-VL-7B-Instruct"
    model_dir: str
    device: str
    torch_dtype: object
    min_tokens: int = 64
    max_tokens: int = 384
    local_files_only: bool = True

    @property
    def min_pixels(self) -> int:
        return self.min_tokens * QWEN_PIXELS_PER_TOKEN

    @property
    def max_pixels(self) -> int:
        return self.max_tokens * QWEN_PIXELS_PER_TOKEN


def _pick_device_and_dtype(torch, device: Optional[str], dtype) -> tuple[str, object]:
    chosen_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is not None:
        return chosen_device, dtype
    if chosen_device == "cuda":
        return chosen_device, torch.float16
    if hasattr(torch.cpu, "is_bf16_supported") and torch.cpu.is_bf16_supported():
        return chosen_device, torch.bfloat16
    return chosen_device, torch.float32


def fit_to_token_budget(image: Image.Image, max_tokens: int) -> Image.Image:
    w, h = image.size
    pixel_budget = max_tokens * QWEN_PIXELS_PER_TOKEN
    if w * h <= pixel_budget:
        return image
    scale = math.sqrt(pixel_budget / (w * h))
    nw = max(QWEN_PATCH, int(w * scale) // QWEN_PATCH * QWEN_PATCH)
    nh = max(QWEN_PATCH, int(h * scale) // QWEN_PATCH * QWEN_PATCH)
    return image.resize((nw, nh), Image.LANCZOS)


# ---- default model key -----------------------------------------------------


def _resolve_vision_model_key(model_key: Optional[str]) -> str:
    """Validate the key exists in VISION_MODELS_REGISTRY, fall back to default."""
    key = model_key or DEFAULT_VISION_MODEL
    if key not in VISION_MODELS_REGISTRY:
        available = list(VISION_MODELS_REGISTRY.keys())
        raise KeyError(
            f"Unknown vision model key {key!r}. "
            f"Available: {available}"
        )
    return key


def build_config(
    model_key: Optional[str] = None,
    device: Optional[str] = None,
    torch_dtype=None,
    min_tokens: int = 64,
    max_tokens: int = 384,
) -> VisionCoderConfig:
    import os.path as osp
    torch = require("torch", reason="VisionCoder requires PyTorch")
    chosen_device, chosen_dtype = _pick_device_and_dtype(torch, device, torch_dtype)

    key = _resolve_vision_model_key(model_key)
    model_dir = get_model_path(model_key)

    
    # Catch the hub-id fallback before transformers tries to go online
    if not osp.isdir(model_dir):
        raise FileNotFoundError(
            f"Vision model {key!r} does not appear to be downloaded locally.\n"
            f"  Expected a directory at: {model_dir}\n"
            f"  DEFAULT_PATHS resolved to: {model_dir!r}\n"
            f"  Run ensure_model({key!r}) or set MODELS_HOME / MODEL_{key.upper().replace('-','_')} "
            f"to point at the correct local path."
        )

    return VisionCoderConfig(
        model_key=key,
        model_dir=model_dir,
        device=chosen_device,
        torch_dtype=chosen_dtype,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )



# ---- the loaded model object -----------------------------------------------

class VisionCoder:
    def __init__(self, cfg: VisionCoderConfig):
        require("transformers", reason="VisionCoder requires HuggingFace transformers")
        self.cfg = cfg
        logger.info(
            "VisionCoder loading key=%s model=%s device=%s dtype=%s token_budget=[%d,%d]",
            cfg.model_key, cfg.model_dir, cfg.device, cfg.torch_dtype,
            cfg.min_tokens, cfg.max_tokens,
        )

        Qwen2_5_VLForConditionalGeneration = get_transformers("Qwen2_5_VLForConditionalGeneration")
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
                do_sample=False,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated = output_ids[:, prompt_len:]
        return self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]

    def analyze_pil(
        self,
        image: Image.Image,
        prompt: str = "Analyze this image.",
        max_new_tokens: int = 128,
        max_tokens: Optional[int] = None,
    ) -> str:
        torch = get_torch()
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
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )
        prompt_len = inputs["input_ids"].shape[1]
        generated = output_ids[:, prompt_len:]
        return self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]

    def analyze_image(
        self,
        image_path: str,
        prompt: str = "Analyze this image.",
        max_new_tokens: int = 128,
        max_tokens: Optional[int] = None,
    ) -> str:
        path = _coerce_image_path(image_path)
        image = Image.open(path).convert("RGB")
        return self.analyze_pil(image, prompt, max_new_tokens, max_tokens)


# ---- per-key instance cache ------------------------------------------------

_INSTANCES: dict[str, VisionCoder] = {}


def get_vision_coder(
    model_key: Optional[str] = None,
    torch_dtype=None,
    max_tokens: int = 384,
    min_tokens: int = 64,
) -> VisionCoder:
    """Return a cached VisionCoder for the given registry key.
    Different keys get different instances; same key is built once."""
    key = _resolve_vision_model_key(model_key)
    if key not in _INSTANCES:
        cfg = build_config(
            model_key=key,
            torch_dtype=torch_dtype,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        )
        _INSTANCES[key] = VisionCoder(cfg)
    return _INSTANCES[key]


# ---- legacy entry point ----------------------------------------------------

def deepcoder_image_analysis(
    image_path,
    prompt: str = "please describe this image",
    max_new_tokens: int = 100,
    model_key: Optional[str] = None,   # was: module_path
    torch_dtype=None,
    max_tokens: Optional[int] = None,
):
    vision = get_vision_coder(model_key=model_key, torch_dtype=torch_dtype)
    return vision.analyze_image(
        image_path=image_path,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        max_tokens=max_tokens,
    )
