## ZeroSearch/ZeroSearch Persistent Manager (Server-optimized)
from ..imports import *


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
_DEFAULT_PATH = DEFAULT_PATHS.get("zerosearch")




# --------------------------------------------------------------------------
# ZeroSearch Persistent Manager
# --------------------------------------------------------------------------
class ZeroSearch(metaclass=SingletonMeta):

    def __init__(self, model_dir: str = None, use_quantization: bool = False):
        if not hasattr(self, "initialized"):
            self.initialized = True
            env = TorchEnvManager()
            self.torch = env.torch
            self.device = env.device
            self.dtype = env.dtype
            self.use_quantization = use_quantization or env.use_quantization
            self.model_dir = self.resolve_model_path(model_dir or _DEFAULT_PATH)
            logger.info(f"DeepCoder using model_dir: {self.model_dir}")
            # ✅ FIX: Resolve the actual path string
         

            self.model = None
            self.tokenizer = None
            self.pipeline = None
            self.lock = threading.Lock()

            logger.info(f"ZeroSearch starting on {self.device} ({self.dtype}) [quantized={self.use_quantization}]")
            self._preload_async()



    def resolve_model_path(entry):
        """Return a valid model path or HF repo id from DEFAULT_PATHS entry."""
        if entry is None:
            logger.error("ZeroSearch: DEFAULT_PATHS entry missing.")
            return None

        if isinstance(entry, dict):
            local_path = entry.get("path")
            repo_id = entry.get("id")

            if local_path and os.path.exists(local_path):
                logger.info(f"ZeroSearch resolved local model path: {local_path}")
                return local_path

            if repo_id:
                logger.info(f"ZeroSearch resolved remote repo id: {repo_id}")
                return repo_id

            logger.error(f"ZeroSearch: malformed entry: {entry}")
            return None

        if isinstance(entry, str):
            logger.info(f"ZeroSearch using direct model string: {entry}")
            return entry

        logger.error(f"ZeroSearch: invalid type for model path: {type(entry)}")
        return None

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
        logger.info(f"Loading ZeroSearch model from {self.model_dir}...")
        kwargs = {"torch_dtype": self.dtype, "device_map": "auto" if "cuda" in self.device else None}

        if self.use_quantization and "cuda" in self.device:
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
                logger.info("Using 4-bit quantization.")
            except ImportError:
                logger.warning("bitsandbytes not installed; skipping quantization.")

        model = get_AutoModelForCausalLM().from_pretrained(self.model_dir, **kwargs)
        model.to(self.device)
        return model
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
