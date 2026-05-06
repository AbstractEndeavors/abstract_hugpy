from .imports import (
    os,
    get_torch,
    get_transformers,
    SingletonMeta,
    get_logFile,
    require,
    Optional,
    DEFAULT_PATHS,
)

from PIL import Image
from .utils import *
logger = get_logFile("vision_coder")


def resolve_qwen_vl_path(module_path: Optional[str] = None) -> str:
    """
    Resolve Qwen-VL path at call time, not import time.

    Priority:
    1. Explicit module_path
    2. MODEL_QWEN_VL env var / DEFAULT_PATHS resolver
    3. Known installed local path fallback
    """
    if module_path:
        return str(module_path)

    resolved = DEFAULT_PATHS.get("qwen_vl")

    if resolved and os.path.exists(str(resolved)):
        return str(resolved)

    fallback = "/var/www/hugging_face/modules/Qwen/Qwen2.5-VL-7B-Instruct"

    if os.path.exists(fallback):
        return fallback

    raise FileNotFoundError(
        "Could not resolve local Qwen2.5-VL model path. "
        "Set MODEL_QWEN_VL=/var/www/hugging_face/modules/Qwen/Qwen2.5-VL-7B-Instruct"
    )


class VisionCoder(metaclass=SingletonMeta):
    """
    Vision-language manager for local Qwen2.5-VL image analysis.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype=None,
        min_tokens: int = 64,
        max_tokens: int = 384,   # << your real lever; raise only if quality drops
    ):
        if hasattr(self, "initialized"):
            return

        torch = require("torch", reason="VisionCoder requires PyTorch")
        require("transformers", reason="VisionCoder requires HuggingFace transformers")

        self.initialized = True
        self.model_dir = resolve_qwen_vl_path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if torch_dtype is None:
            self.torch_dtype = torch.float16 if self.device == "cuda" else pick_cpu_dtype(torch)
        else:
            self.torch_dtype = torch_dtype

        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.min_pixels = min_tokens * QWEN_TOKEN_BYTES
        self.max_pixels = max_tokens * QWEN_TOKEN_BYTES

        logger.info(
            f"VisionCoder model={self.model_dir} device={self.device} "
            f"dtype={self.torch_dtype} token_budget=[{min_tokens},{max_tokens}]"
        )


    def _load_model(self):
        Qwen2_5_VLForConditionalGeneration = get_transformers(
            "Qwen2_5_VLForConditionalGeneration"
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_dir,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )

        self.model.eval()


    def _load_processor(self):
        AutoProcessor = get_transformers("AutoProcessor")
        self.processor = AutoProcessor.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
            local_files_only=True,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

    def analyze_image(
        self,
        image_path: str,
        prompt: str = "Analyze this image.",
        max_new_tokens: int = 1000,
    ) -> str:
        torch = get_torch()

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image = fit_to_token_budget(image, self.max_tokens)  # belt-and-braces with processor cap

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )

        target_device = next(self.model.parameters()).device
        inputs = {
            key: value.to(target_device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]

        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


def get_vision_coder(
    module_path: Optional[str] = None,
    torch_dtype=None,
) -> VisionCoder:
    model_path = resolve_qwen_vl_path(module_path)

    logger.info(f"Resolved qwen_vl model path: {model_path}")

    return VisionCoder(
        model_dir=model_path,
        torch_dtype=torch_dtype,
    )


def deepcoder_image_analysis(
    image_path,
    prompt="please describe this image",
    max_new_tokens: int = 100,
    module_path: Optional[str] = None,
    torch_dtype=None,
):

    vision = get_vision_coder(
        module_path=module_path,
        torch_dtype=torch_dtype,
    )

    return vision.analyze_image(
        image_path=image_path,
        prompt=prompt,
        max_new_tokens=max_new_tokens
    )
