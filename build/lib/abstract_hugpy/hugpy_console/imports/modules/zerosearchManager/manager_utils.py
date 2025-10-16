## DeepCoder/ZeroSearch Persistent Manager (Server-optimized)
```python
import os, logging, threading
from typing import Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor

from abstract_utilities import SingletonMeta
from .imports import (
    get_torch,
    get_pipeline,
    get_AutoTokenizer,
    get_AutoModelForCausalLM,
    get_GenerationConfig,
)
from .constants import DEFAULT_PATHS, MODULE_DEFAULTS


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("zerosearch_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ZeroSearch")


# --------------------------------------------------------------------------
# Torch env resolver (cached)
# --------------------------------------------------------------------------
_TORCH_ENV = None

def get_torch_env():
    global _TORCH_ENV
    if _TORCH_ENV is not None:
        return _TORCH_ENV

    torch = get_torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    _TORCH_ENV = (torch, device, dtype)
    logger.info(f"Torch env: device={device}, dtype={dtype}")
    return _TORCH_ENV


# --------------------------------------------------------------------------
# ZeroSearch Persistent Manager
# --------------------------------------------------------------------------
class ZeroSearch(metaclass=SingletonMeta):
    """Persistent DeepCoder model interface optimized for long-running services."""

    def __init__(self, model_dir: str = None, use_quantization: bool = False):
        if hasattr(self, "initialized"):
            return

        self.initialized = True
        self.model_dir = model_dir or DEFAULT_PATHS.get("deepcoder")
        self.use_quantization = use_quantization
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.lock = threading.Lock()

        torch, device, dtype = get_torch_env()
        self.torch, self.device, self.dtype = torch, device, dtype

        logger.info(f"Initializing ZeroSearch on {self.device} ({self.dtype})...")
        self._preload_async()

    # ------------------------------------------------------------------
    # Background preload
    # ------------------------------------------------------------------
    def _preload_async(self):
        """Preload model/tokenizer in a background thread to avoid blocking server startup."""
        thread = threading.Thread(target=self._safe_preload, daemon=True)
        thread.start()

    def _safe_preload(self):
        try:
            self._load_model_and_tokenizer()
            logger.info("ZeroSearch model preloaded successfully.")
        except Exception as e:
            logger.error(f"Preload failed: {e}")

    # ------------------------------------------------------------------
    # Loading utilities
    # ------------------------------------------------------------------
    def _load_model_and_tokenizer(self):
        """Parallelized load of model and tokenizer."""
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_model = ex.submit(self._load_model)
            f_tok = ex.submit(self._load_tokenizer)
            self.model = f_model.result()
            self.tokenizer = f_tok.result()
        self._load_generation_config()
        self._create_pipeline()

    def _load_model(self):
        torch = self.torch
        logger.info(f"Loading DeepCoder model from {self.model_dir}...")
        kwargs = {"torch_dtype": self.dtype, "device_map": "auto" if self.device == "cuda" else None}

        if self.use_quantization and self.device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
                logger.info("Using 4-bit quantization.")
            except ImportError:
                logger.warning("bitsandbytes not installed; skipping quantization.")

        return get_AutoModelForCausalLM().from_pretrained(self.model_dir, **kwargs)

    def _load_tokenizer(self):
        tok = get_AutoTokenizer().from_pretrained(self.model_dir, trust_remote_code=True)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
            logger.info("Set pad_token_id to eos_token_id.")
        return tok

    def _load_generation_config(self):
        try:
            self.generation_config = get_GenerationConfig().from_pretrained(self.model_dir)
            logger.info("Generation config loaded successfully.")
        except Exception as e:
            logger.warning(f"Using default generation config ({e}).")
            self.generation_config = None

    def _create_pipeline(self):
        """Create and cache the generation pipeline."""
        if self.pipeline is not None:
            return
        self.pipeline = get_pipeline()(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
        )
        logger.info("ZeroSearch text-generation pipeline initialized.")

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.95,
        use_chat_template: bool = False,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Thread-safe text generation call."""
        with self.lock:
            if self.model is None or self.tokenizer is None:
                logger.info("Lazy-loading model due to first request...")
                self._load_model_and_tokenizer()

            if use_chat_template and messages:
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            else:
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            try:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return result.strip()

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.error("OOM detected. Clearing CUDA cache and retrying on CPU...")
                    self._recover_to_cpu()
                    return self.generate(prompt, max_new_tokens, temperature, top_p)
                else:
                    raise

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------
    def _recover_to_cpu(self):
        """Fallback if GPU fails."""
        torch = self.torch
        torch.cuda.empty_cache()
        self.device = "cpu"
        self.model.to("cpu")
        logger.warning("Model moved to CPU due to GPU memory constraints.")

    # ------------------------------------------------------------------
    # Info & housekeeping
    # ------------------------------------------------------------------
    def get_info(self) -> Dict[str, Union[str, int]]:
        return {
            "model_dir": self.model_dir,
            "device": self.device,
            "dtype": str(self.dtype),
            "quantized": self.use_quantization,
            "initialized": self.model is not None,
        }
