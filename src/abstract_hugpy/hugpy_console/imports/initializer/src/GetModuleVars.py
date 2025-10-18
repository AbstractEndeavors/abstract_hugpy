import os
import logging
from typing import *
from typing import Optional, Any, Dict, Tuple
from .ModelHubLoader import ModelHubLoader
# If you have a custom logger helper, we’ll use it if present.
try:
    from your_logging_helpers import get_logFile  # optional
except Exception:
    get_logFile = (__name__)

# Reuse the loader you already have (paste it above or import it)
# from your_loader_module import ModelHubLoader
MODULE_DEFAULTS = {
    "whisper": {
        "path": "/mnt/24T/hugging_face/modules/whisper_base",
        "repo_id": "openai/whisper-base",
        "handle":"whisper"
    },
    "keybert": {
        "path": "/mnt/24T/hugging_face/modules/all_minilm_l6_v2",
        "repo_id": "sentence-transformers/all-MiniLM-L6-v2",
        "handle": "keybert"
    },
    "summarizer": {
        "path": "/mnt/24T/hugging_face/modules/text_summarization",
        "repo_id": "Falconsai/text_summarization",
        "handle": "summarizer"
    },
    "flan": {
        "path": "/mnt/24T/hugging_face/modules/flan_t5_xl",
        "repo_id": "google/flan-t5-xl",
        "handle": "flan"
    },
    "bigbird": {
        "path": "/mnt/24T/hugging_face/modules/led_large_16384",
        "repo_id": "allenai/led-large-16384",
        "handle": "bigbird"
    },
    "deepcoder": {
        "path": "/mnt/24T/hugging_face/modules/DeepCoder-14B",
        "repo_id": "agentica-org/DeepCoder-14B-Preview",
        "handle": "deepcoder"
    },
    "huggingface": {
        "path": "/mnt/24T/hugging_face/modules/hugging_face_models",
        "repo_id": "huggingface/hub",
        "handle": "hugging_face_models"
    },
    "zerosearch": {
        "path": "/mnt/24T/hugging_face/modules/ZeroSearch_dataset",
        "repo_id": "ZeroSearch/dataset",
        "handle": "ZeroSearch"
    }
}
class GetModuleVars(metaclass=type):
    """
    Robust, singleton-ish module wrapper that:
      • resolves a model source from a module name or explicit string
      • prefers local dir if present, else falls back to HF repo id
      • safely loads AutoTokenizer / AutoModelForCausalLM with optional 4-bit quantization
      • auto-selects device (cuda/cpu) and dtype
      • exposes simple .generate()

    Parameters
    ----------
    name : str
        A key from your defaults dict (e.g. "deepcoder"). Ignored if `source` is provided.
    source : str | None
        Explicit local dir or "namespace/repo". If provided, overrides `name`.
    cache_dir : str | None
        HF cache dir. If None, defaults to HF’s global cache.
    is_cuda : bool | None
        Force CUDA availability flag. If None, auto-detect.
    device : str | None
        Force device string. If None, use "cuda" if available else "cpu".
    use_fast : bool
        Whether to prefer fast tokenizers.
    trust_remote_code : bool
        Forwarded to HF loaders.
    use_quantization : bool
        If True and on CUDA, tries to load in 4-bit (bitsandbytes).
    torch_dtype : Any | "auto" | None
        Target dtype. "auto" picks bfloat16 on CUDA, else float32 on CPU.
    device_map : Any | None
        Optional accelerate/transformers device map; "auto" is common when quantizing.
    prefer_local : bool
        Prefer local directory in defaults over remote repo id.
    must_be_transformers_dir : bool
        If True, enforce that local dir has a config.json.
    defaults : dict | None
        Your DEFAULT_PATHS/MODULE_DEFAULTS-shaped mapping.
    loader : ModelHubLoader | None
        Custom loader instance; if None, a new one is created with `defaults`.

    Attributes
    ----------
    tokenizer, model, generation_config
    """

    _instance = None  # very light singleton (optional). Remove if you truly want multi-instances.

    def __new__(cls, *args, **kwargs):
        # feel free to drop singleton if you want multiple parallel modules
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        source: Optional[str] = None,
        is_cuda: Optional[bool] = None,
        device: Optional[str] = None,
        use_fast: bool = True,
        trust_remote_code: bool = True,
        use_quantization: bool = False,
        torch_dtype: Optional[Any] = None,
        device_map: Optional[Any] = None,
        prefer_local: bool = True,
        must_be_transformers_dir: bool = False,
        defaults: Optional[Dict[str, Dict[str, str]]] = None,
        loader: Optional["ModelHubLoader"] = None,
    ):
        if getattr(self, "_initialized", False):
            return

        # logger
        self.logger = get_logFile() if callable(get_logFile or (lambda: None)) else logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)

        self.name = name or "deepcoder"
        self.cache_dir = cache_dir
        self.source_arg = source
        self.use_fast = bool(use_fast)
        self.trust_remote_code = bool(trust_remote_code)
        self.use_quantization = bool(use_quantization)
        self.torch_dtype = torch_dtype  # may be "auto"
        self.device_map = device_map

        # loader / defaults
        self.loader = loader or ModelHubLoader(defaults=MODULE_DEFAULTS or {})
        if defaults:
            self.loader.set_defaults(defaults)

        # torch module (lazy via loader)
        self.torch = self.loader.torch()

        # device selection
        self.is_cuda = bool(is_cuda) if is_cuda is not None else self.torch.cuda.is_available()
        self.device = device or ("cuda" if self.is_cuda else "cpu")

        # resolve source (string only!)
        self.model_dir = self._resolve_source(
            name=self.name,
            source=self.source_arg,
            prefer_local=prefer_local,
            must_be_transformers_dir=must_be_transformers_dir,
        )

        # dtype selection
        self.dtype = self._pick_dtype(self.torch_dtype)

        # actual loads
        self.tokenizer = None
        self.model = None
        self.generation_config = None

        self._load_tokenizer()
        self._load_model()
        self._load_generation_config()

        self._initialized = True
        self.logger.info("Module initialized successfully.")

    # ---------- helpers ----------
    def _resolve_source(
        self,
        name: str,
        source: Optional[str],
        prefer_local: bool,
        must_be_transformers_dir: bool,
    ) -> str:
        if source and isinstance(source, str):
            return self.loader._guard_src_for_from_pretrained(source)
        # resolve from defaults by name
        return self.loader.resolve_src(
            name,
            prefer_local=prefer_local,
            require_exists=False,
            must_be_transformers_dir=must_be_transformers_dir,
        )

    def _pick_dtype(self, torch_dtype: Optional[Any]) -> Any:
        if torch_dtype == "auto" or torch_dtype is None:
            if self.device == "cuda":
                # prefer bf16 on modern GPUs; fallback to fp16 if needed
                return getattr(self.torch, "bfloat16", self.torch.float16)
            return self.torch.float32
        # If user passed a string like "bfloat16"
        if isinstance(torch_dtype, str):
            return getattr(self.torch, torch_dtype)
        return torch_dtype

    # ---------- loads ----------
    def _load_tokenizer(self):
        AutoTokenizer = self.loader.AutoTokenizer()
        self.logger.info(f"Loading tokenizer from {self.model_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            cache_dir=self.cache_dir,
            use_fast=self.use_fast,
            trust_remote_code=self.trust_remote_code,
        )
        # Ensure pad token
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if self.tokenizer.pad_token_id is None and getattr(self.tokenizer, "unk_token_id", None) is not None:
                self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        self.logger.info("Tokenizer loaded.")

    def _maybe_quant_config(self):
        if not self.use_quantization or self.device != "cuda":
            return {}
        try:
            # transformers>=4.36
            BitsAndBytesConfig = getattr(self.loader.transformers(), "BitsAndBytesConfig")
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=getattr(self.torch, "bfloat16", self.torch.float16),
                bnb_4bit_quant_type="nf4",
            )
            # if user didn’t set device_map, auto is sensible with quantized load
            dm = self.device_map if self.device_map is not None else "auto"
            return {"quantization_config": quant_cfg, "device_map": dm}
        except Exception as e:
            self.logger.warning(f"4-bit quantization not available ({e}); loading full precision instead.")
            return {}

    def _load_model(self):
        AutoModelForCausalLM = self.loader.AutoModelForCausalLM()
        self.logger.info(f"Loading model from {self.model_dir}...")

        extra = self._maybe_quant_config()
        model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            cache_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=self.dtype,
            **extra,
        )

        # If not using a device_map that already places on cuda, move explicitly
        if not extra.get("device_map"):
            model = model.to(self.device)

        self.model = model
        self.logger.info("Model loaded.")

    def _load_generation_config(self):
        try:
            GenerationConfig = self.loader.GenerationConfig()
            # try to load from repo if present; fallback to a sensible default
            try:
                self.generation_config = GenerationConfig.from_pretrained(
                    self.model_dir, cache_dir=self.cache_dir, trust_remote_code=self.trust_remote_code
                )
            except Exception:
                self.generation_config = GenerationConfig(
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
                )
            self.logger.info("Generation config ready.")
        except Exception:
            self.generation_config = None

    # ---------- public API ----------
    @property
    def model_sources(self) -> Dict[str, str]:
        """Quick peek at resolved values."""
        return {
            "model_dir": self.model_dir,
            "cache_dir": self.cache_dir,
            "device": self.device,
            "dtype": str(self.dtype),
            "quantized": self.use_quantization and self.device == "cuda",
        }

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1000,
        temperature: float = 0.6,
        top_p: float = 0.95,
        use_chat_template: bool = False,
        messages: Optional[List[Dict[str, str]]] = None,
        do_sample: bool = False,
        **kwargs
    ) -> str:
        """
        Generate text based on the input prompt or chat messages.
        NOTE: Only pass *generation* kwargs to model.generate. Filter out loader-only flags.
        """

        # 1) Prepare inputs with/without chat template (handled BEFORE generate)
        if use_chat_template and messages:
            if hasattr(self.tokenizer, "apply_chat_template"):
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            else:
                # Fallback: naive stitch if tokenizer has no chat template
                stitched = ""
                for m in messages:
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    stitched += f"{role}: {content}\n"
                stitched += "assistant: "
                inputs = self.tokenizer(stitched, return_tensors="pt", padding=True)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 2) Build clean generation kwargs (strip loader/config flags)
        allowed_gen_keys = {
            "max_new_tokens","min_new_tokens",
            "temperature","top_p","top_k",
            "num_beams","length_penalty",
            "repetition_penalty","no_repeat_ngram_size",
            "do_sample","early_stopping",
            "eos_token_id","pad_token_id","bos_token_id",
            "num_return_sequences","return_dict_in_generate",
            "output_scores","use_cache"
        }

        gen_kwargs = {k: v for k, v in kwargs.items() if k in allowed_gen_keys}
        gen_kwargs.setdefault("max_new_tokens", max_new_tokens)
        gen_kwargs.setdefault("temperature", temperature)
        gen_kwargs.setdefault("top_p", top_p)
        gen_kwargs.setdefault("do_sample", do_sample)
        gen_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        gen_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)

        # 3) Optional: silence warnings when do_sample=False but temp/top_p set
        if not gen_kwargs.get("do_sample", False):
            gen_kwargs.pop("temperature", None)
            gen_kwargs.pop("top_p", None)

        # 4) Generate
        with self.torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # 5) Decode only the generated continuation
        # If your tokenizer inserts prompt tokens, decode the new tokens:
        # (simple path: decode full and return)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text
