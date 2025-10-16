from ..imports import *


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
logger = get_logFile("zerosearch")

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

    logger.error(f"ZeroSearch: invalid model path type: {type(entry)}")
    return None


# --------------------------------------------------------------------------
# ZeroSearch Persistent Manager
# --------------------------------------------------------------------------
class ZeroSearch(BaseModelManager):
    """Persistent ZeroSearch model interface optimized for long-running inference."""

    def __init__(self, model_dir: str = None, use_quantization: bool = False,**kwargs):
        if not hasattr(self, "initialized"):
           
            self.model_dir = model_dir or DEFAULT_PATHS.get("zerosearch")

            # ✅ Defensive safeguard
   

            self.initialized = True
            env = TorchEnvManager()
            self.torch = env.torch
            self.device = env.device
            self.dtype = env.dtype
            self.use_quantization = use_quantization or env.use_quantization
            self.model_dir = resolve_model_path(
                    model_dir or DEFAULT_PATHS.get("zerosearch")
                )
            self.model = None
            self.tokenizer = None
            self.pipeline = None
            self.generation_config = None
            self.lock = threading.Lock()

            logger.info(
                f"ZeroSearch initializing on {self.device} ({self.dtype}) [quantized={self.use_quantization}]"
            )
            self._preload_async()

    # ------------------------------------------------------------------
    # Async preload
    # ------------------------------------------------------------------
    def _preload_async(self):
        """Preload model/tokenizer asynchronously at startup."""
        thread = threading.Thread(target=self._safe_preload, daemon=True)
        thread.start()

    def _safe_preload(self):
        try:
            self._load_model_and_tokenizer()
            logger.info("ZeroSearch model preloaded successfully.")
        except Exception as e:
            logger.error(f"ZeroSearch preload failed: {e}")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model_and_tokenizer(self):
        """Parallelized model/tokenizer load."""
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_model = ex.submit(self._load_model)
            f_tok = ex.submit(self._load_tokenizer)
            self.model = f_model.result()
            self.tokenizer = f_tok.result()

        self._load_generation_config()
        self._create_pipeline()

    def _load_model(self):
        logger.info(f"Loading ZeroSearch model from {self.model_dir}...")
        AutoModelForCausalLM = get_AutoModelForCausalLM()

        kwargs = {"torch_dtype": self.dtype}
        if "cuda" in self.device:
            kwargs["device_map"] = "auto"

        if self.use_quantization and "cuda" in self.device:
            try:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
                logger.info("Using 4-bit quantization.")
            except ImportError:
                logger.warning("bitsandbytes not installed; skipping quantization.")

        model = AutoModelForCausalLM.from_pretrained(self.model_dir, **kwargs)
        model.to(self.device)
        return model

    def _load_tokenizer(self):
        AutoTokenizer = get_AutoTokenizer()
        tok = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
            logger.info("Set pad_token_id to eos_token_id.")
        return tok

    def _load_generation_config(self):
        GenerationConfig = get_GenerationConfig()
        try:
            self.generation_config = GenerationConfig.from_pretrained(self.model_dir)
            logger.info("Generation config loaded successfully.")
        except Exception as e:
            logger.warning(f"Using default generation config ({e}).")
            self.generation_config = GenerationConfig(
                do_sample=True, temperature=0.6, top_p=0.95, max_new_tokens=64000
            )

    def _create_pipeline(self):
        if self.pipeline is not None:
            return
        pipeline = get_pipeline()
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if "cuda" in self.device else -1,
        )
        logger.info("ZeroSearch text-generation pipeline initialized.")

    # ------------------------------------------------------------------
    # Core generation API
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
        """Thread-safe lazy generation."""
        with self.lock:
            if self.model is None or self.tokenizer is None:
                logger.info("Lazy-loading ZeroSearch model on first request...")
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
                    logger.error("OOM detected; retrying on CPU...")
                    self._recover_to_cpu()
                    return self.generate(prompt, max_new_tokens, temperature, top_p)
                else:
                    raise

    # ------------------------------------------------------------------
    # Recovery & Info
    # ------------------------------------------------------------------
    def _recover_to_cpu(self):
        self.torch.cuda.empty_cache()
        self.device = "cpu"
        self.model.to("cpu")
        logger.warning("ZeroSearch model moved to CPU due to GPU memory constraints.")

    def get_info(self) -> Dict[str, Union[str, int]]:
        return {
            "model_name": "ZeroSearch",
            "model_dir": self.model_dir,
            "device": self.device,
            "dtype": str(self.dtype),
            "quantized": self.use_quantization,
            "initialized": self.model is not None,
        }


