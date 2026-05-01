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

DEFAULT_VISION_PATH = DEFAULT_PATHS.get("qwen_vl")
logger = get_logFile("vision_coder")


class VisionCoder(metaclass=SingletonMeta):
    """
    Vision-language manager for local Qwen2.5-VL image analysis.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype=None,
    ):
        if hasattr(self, "initialized"):
            return

        torch = require("torch", reason="VisionCoder requires PyTorch")
        require("transformers", reason="VisionCoder requires HuggingFace transformers")

        self.initialized = True
        self.model_dir = model_dir or DEFAULT_VISION_PATH
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or torch.float16

        self.model = None
        self.processor = None

        self._load_model()
        self._load_processor()

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
    return VisionCoder(
        model_dir=module_path or DEFAULT_PATHS["qwen_vl"],
        torch_dtype=torch_dtype,
    )
