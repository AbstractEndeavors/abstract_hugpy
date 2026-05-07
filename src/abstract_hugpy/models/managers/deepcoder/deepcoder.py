from .imports import (
    os,
    get_torch,
    get_transformers,
    SingletonMeta,
    get_logFile,
    require,
    DEFAULT_PATHS,
    Dict,
    List,
    Optional,
    Union,
)

DEFAULT_PATH: str = DEFAULT_PATHS.get("deepcoder")
logger = get_logFile("deepcoder")


class DeepCoder(metaclass=SingletonMeta):
    """A robust Python module for interacting with the DeepCoder-14B-Preview model."""

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype=None,
        use_quantization: bool = False,
        refresh_model: bool = False,
    ):
        model_dir = model_dir or DEFAULT_PATH

        if not model_dir:
            raise ValueError("DeepCoder requires a model_dir or DEFAULT_PATHS['deepcoder'].")

        torch = require("torch", reason="DeepCoder requires PyTorch")
        require("transformers", reason="DeepCoder requires HuggingFace transformers")

        requested_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        requested_dtype = torch_dtype or torch.float16

        same_config = (
            hasattr(self, "initialized")
            and getattr(self, "model_dir", None) == model_dir
            and getattr(self, "device", None) == requested_device
            and getattr(self, "torch_dtype", None) == requested_dtype
            and getattr(self, "use_quantization", None) == use_quantization
        )

        if hasattr(self, "initialized") and same_config and not refresh_model:
            logger.info("DeepCoder already initialized with same config; reusing model.")
            return

        if hasattr(self, "initialized") and refresh_model:
            logger.info("Refreshing DeepCoder model by request.")
            self.unload_model()

        elif hasattr(self, "initialized") and not same_config:
            logger.info("DeepCoder config changed; reloading model.")
            self.unload_model()

        self.initialized = True
        self.model_dir = model_dir
        self.device = requested_device
        self.torch_dtype = requested_dtype
        self.use_quantization = use_quantization

        self.model = None
        self.tokenizer = None
        self.generation_config = None

        try:
            self._load_model()
            self._load_tokenizer()
            self._load_generation_config()
            logger.info("DeepCoder module initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize DeepCoder: {str(e)}")
            self.initialized = False
            raise

    def unload_model(self):
        """Release model/tokenizer references and clear CUDA cache when available."""
        torch = get_torch()

        for attr in ("model", "tokenizer", "generation_config"):
            if hasattr(self, attr):
                setattr(self, attr, None)

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                logger.info("CUDA cache cleared after DeepCoder unload.")
        except Exception as e:
            logger.warning(f"CUDA cache cleanup failed: {repr(e)}")

    def _load_model(self):
        AutoModelForCausalLM = get_transformers("AutoModelForCausalLM")

        kwargs = {
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": True,
        }

        if self.use_quantization and self.device == "cuda":
            try:
                BitsAndBytesConfig = get_transformers("BitsAndBytesConfig")
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            except (ImportError, KeyError):
                logger.warning("BitsAndBytesConfig unavailable — skipping quantization.")
                self.use_quantization = False

        logger.info(f"Loading DeepCoder model from {self.model_dir} on {self.device}...")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            **kwargs,
        )

        if not self.use_quantization:
            self.model = self.model.to(self.device)

        self.model.eval()

    def _load_tokenizer(self):
        AutoTokenizer = get_transformers("AutoTokenizer")

        logger.info(f"Loading tokenizer from {self.model_dir}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.info("Set pad_token_id to eos_token_id.")

        logger.info("Tokenizer loaded successfully.")

    def _load_generation_config(self):
        GenerationConfig = get_transformers("GenerationConfig")

        try:
            self.generation_config = GenerationConfig.from_pretrained(self.model_dir)
            logger.info("Generation configuration loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load generation config: {str(e)}")

            self.generation_config = GenerationConfig(
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                max_new_tokens=64000,
            )

            logger.info("Using default generation configuration.")

    def generate(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_new_tokens: int = 1000,
        temperature: float = 0.6,
        top_p: float = 0.95,
        use_chat_template: bool = False,
        messages: Optional[List[Dict[str, str]]] = None,
        do_sample: bool = False,
        *args,
        **kwargs,
    ) -> str:
        torch = get_torch()

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("DeepCoder model/tokenizer is not loaded.")

        try:
            if use_chat_template:
                chat_messages = messages

                if chat_messages is None:
                    if isinstance(prompt, list):
                        chat_messages = prompt
                    else:
                        chat_messages = [{"role": "user", "content": prompt}]

                inputs = self.tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
            else:
                inputs = self.tokenizer(
                    str(prompt),
                    return_tensors="pt",
                    padding=True,
                )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            config_max = getattr(self.generation_config, "max_new_tokens", None)

            if not isinstance(config_max, int):
                logger.warning(
                    "generation_config.max_new_tokens is not set or invalid. "
                    "Using requested max_new_tokens only."
                )
                config_max = max_new_tokens

            final_max = min(max_new_tokens, config_max)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=final_max,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    *args,
                    **kwargs,
                )

            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
            )

            logger.info("Text generation completed successfully.")
            return generated_text

        except Exception as e:
            logger.error(f"Text generation failed: {repr(e)}")
            raise

    def save_output(self, text: str, output_path: str):
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)

            logger.info(f"Output saved to {output_path}.")
        except Exception as e:
            logger.error(f"Failed to save output: {str(e)}")
            raise

    def get_model_info(self) -> Dict[str, Union[str, int, bool]]:
        return {
            "model_name": "DeepCoder-14B-Preview",
            "architecture": "Qwen2ForCausalLM",
            "num_layers": 48,
            "hidden_size": 5120,
            "vocab_size": 152064,
            "model_dir": self.model_dir,
            "device": self.device,
            "torch_dtype": str(self.torch_dtype),
            "quantized": self.use_quantization,
            "loaded": self.model is not None,
        }


_deep_coder: Optional[DeepCoder] = None


def get_deep_coder(
    module_path: Optional[str] = None,
    model_dir: Optional[str] = None,
    device: Optional[str] = None,
    torch_dtype=None,
    use_quantization: Optional[bool] = None,
    refresh_model: bool = False,
) -> DeepCoder:
    """
    Public constructor/cache accessor.

    Accepts both module_path and model_dir so older callsites do not break.
    """
    resolved_model_dir = model_dir or module_path or DEFAULT_PATH

    if torch_dtype is None:
        torch_dtype = get_torch().float16

    if use_quantization is None:
        use_quantization = True

    return DeepCoder(
        model_dir=resolved_model_dir,
        device=device,
        torch_dtype=torch_dtype,
        use_quantization=use_quantization,
        refresh_model=refresh_model,
    )


def _get_model(
    model_dir: Optional[str] = None,
    device: Optional[str] = None,
    torch_dtype=None,
    use_quantization: bool = False,
    refresh_model: bool = False,
) -> DeepCoder:
    global _deep_coder

    resolved_model_dir = model_dir or DEFAULT_PATH

    if _deep_coder is None:
        _deep_coder = get_deep_coder(
            model_dir=resolved_model_dir,
            device=device,
            torch_dtype=torch_dtype,
            use_quantization=use_quantization,
            refresh_model=False,
        )
        return _deep_coder

    existing_config_changed = (
        getattr(_deep_coder, "model_dir", None) != resolved_model_dir
        or getattr(_deep_coder, "device", None) != (
            device or getattr(_deep_coder, "device", None)
        )
        or getattr(_deep_coder, "use_quantization", None) != use_quantization
        or (
            torch_dtype is not None
            and getattr(_deep_coder, "torch_dtype", None) != torch_dtype
        )
    )

    if refresh_model or existing_config_changed:
        _deep_coder = get_deep_coder(
            model_dir=resolved_model_dir,
            device=device,
            torch_dtype=torch_dtype,
            use_quantization=use_quantization,
            refresh_model=True,
        )

    return _deep_coder


def deep_coder_generate(
    prompt: Union[str, List[Dict[str, str]]],
    max_new_tokens: int = 20,
    temperature: float = 0.6,
    top_p: float = 0.95,
    use_chat_template: bool = False,
    messages: Optional[List[Dict[str, str]]] = None,
    do_sample: bool = False,
    model_dir: Optional[str] = None,
    device: Optional[str] = None,
    torch_dtype=None,
    use_quantization: bool = False,
    refresh_model: bool = False,
    *args,
    **kwargs,
) -> str:
    model = _get_model(
        model_dir=model_dir,
        device=device,
        torch_dtype=torch_dtype,
        use_quantization=use_quantization,
        refresh_model=refresh_model,
    )

    return model.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        use_chat_template=use_chat_template,
        messages=messages,
        do_sample=do_sample,
        *args,
        **kwargs,
    )


def refresh_deep_coder(
    model_dir: Optional[str] = None,
    device: Optional[str] = None,
    torch_dtype=None,
    use_quantization: bool = False,
) -> DeepCoder:
    """
    Explicit refresh helper for Flask routes, admin panels, or CLI commands.
    """
    return _get_model(
        model_dir=model_dir,
        device=device,
        torch_dtype=torch_dtype,
        use_quantization=use_quantization,
        refresh_model=True,
    )


def try_deep_coder(
    module_path: Optional[str] = None,
    torch_dtype=None,
    use_quantization: Optional[bool] = None,
    refresh_model: bool = False,
):
    try:
        deepcoder = get_deep_coder(
            module_path=module_path,
            torch_dtype=torch_dtype,
            use_quantization=use_quantization,
            refresh_model=refresh_model,
        )

        logger.info("DeepCoder logger initialized and active.")

        prompt = "Write a Python function to calculate the factorial of a number."

        generated_text = deepcoder.generate(
            prompt=prompt,
            max_new_tokens=2,
            use_chat_template=False,
        )

        print("Generated Text:", generated_text)

        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {
                "role": "user",
                "content": "Explain how to implement a binary search in Python.",
            },
        ]

        chat_response = deepcoder.generate(
            prompt=messages,
            max_new_tokens=1000,
            use_chat_template=True,
        )

        print("Chat Response:", chat_response)

        deepcoder.save_output(
            chat_response,
            "./output/binary_search_explanation.txt",
        )

        print("Model Info:", deepcoder.get_model_info())

    except Exception as e:
        logger.error(f"Example usage failed: {str(e)}")
        raise
