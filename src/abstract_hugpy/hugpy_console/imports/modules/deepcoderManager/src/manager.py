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
logger = get_logFile("deepcoder")
_DEFAULT_PATH = DEFAULT_PATHS.get("deepcoder")


def resolve_model_path(entry):
    """Return a valid model path or HF repo id from DEFAULT_PATHS entry."""
    if entry is None:
        logger.error("DeepCoder: DEFAULT_PATHS entry missing.")
        return None

    if isinstance(entry, dict):
        local_path = entry.get("path")
        repo_id = entry.get("id")

        if local_path and os.path.exists(local_path):
            logger.info(f"DeepCoder resolved local model path: {local_path}")
            return local_path

        if repo_id:
            logger.info(f"DeepCoder resolved remote repo id: {repo_id}")
            return repo_id

        logger.error(f"DeepCoder: malformed entry: {entry}")
        return None

    if isinstance(entry, str):
        logger.info(f"DeepCoder using direct model string: {entry}")
        return entry

    logger.error(f"DeepCoder: invalid model path type: {type(entry)}")
    return None

# --------------------------------------------------------------------------
# ZeroSearch Persistent Manager
# --------------------------------------------------------------------------

class DeepCoder(BaseModelManager):
    """Server-optimized persistent manager for DeepCoder-14B-Preview."""

    def __init__(self, model_dir: str = None, use_quantization: bool = False):
        if not hasattr(self, "initialized"):
            self.initialized = True
            self.torch_env = TorchEnvManager()
            self.torch = self.torch_env.torch
            self.device = self.torch_env.device
            self.dtype = self.torch_env.dtype
            self.use_quantization = self.torch_env.use_quantization
            self.preload()





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

    def _load_model_and_tokenizer(self):
        logger.info(f"Loading DeepCoder model from {self.model_dir}...")
        AutoModelForCausalLM = get_AutoModelForCausalLM()
        AutoTokenizer = get_AutoTokenizer()

        kwargs = {"torch_dtype": self.dtype}
        if "cuda" in self.device:
            kwargs["device_map"] = "auto"

        if self.use_quantization and "cuda" in self.device:
            try:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
                logger.info("Using 4-bit quantization.")
            except ImportError:
                logger.warning("bitsandbytes not installed; skipping quantization.")

        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.info("Set pad_token_id to eos_token_id.")

        logger.info("DeepCoder model and tokenizer loaded successfully.")


    def _load_generation_config(self):
        try:
            self.generation_config = get_GenerationConfig().from_pretrained(self.model_dir)
            logger.info("Generation config loaded successfully.")
        except Exception as e:
            logger.warning(f"Using default generation config ({e}).")
            self.generation_config = None

    def _create_pipeline(self):
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model and tokenizer must be loaded first.")
        self.pipeline = get_pipeline()(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if "cuda" in self.device else -1,
        )
        logger.info("DeepCoder text-generation pipeline initialized.")

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
        """Thread-safe lazy generation wrapper."""
        with self.lock:
            if self.model is None or self.tokenizer is None:
                logger.info("Lazy-loading DeepCoder model due to first request...")
                self._safe_preload()

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
                    logger.error("OOM detected. Moving DeepCoder to CPU...")
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

