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
    """Persistent DeepCoder model manager optimized for Flask/server inference."""

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype=None,
        use_quantization: bool = False,
        refresh_model: bool = False,
        use_flash_attention: bool = False,
        local_files_only: bool = True,
    ):
        model_dir = model_dir or DEFAULT_PATH

        if not model_dir:
            raise ValueError("DeepCoder requires a model_dir or DEFAULT_PATHS['deepcoder'].")

        torch = require("torch", reason="DeepCoder requires PyTorch")
        require("transformers", reason="DeepCoder requires HuggingFace transformers")

        requested_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if torch_dtype is None:
            if requested_device == "cuda" and torch.cuda.is_bf16_supported():
                requested_dtype = torch.bfloat16
            elif requested_device == "cuda":
                requested_dtype = torch.float16
            else:
                requested_dtype = torch.float32
        else:
            requested_dtype = torch_dtype

        same_config = (
            hasattr(self, "initialized")
            and getattr(self, "model_dir", None) == model_dir
            and getattr(self, "device", None) == requested_device
            and getattr(self, "torch_dtype", None) == requested_dtype
            and getattr(self, "use_quantization", None) == use_quantization
            and getattr(self, "use_flash_attention", None) == use_flash_attention
            and getattr(self, "local_files_only", None) == local_files_only
        )

        if hasattr(self, "initialized") and same_config and not refresh_model:
            logger.info("DeepCoder already initialized with same config; reusing model.")
            return

        if hasattr(self, "initialized"):
            logger.info("DeepCoder config changed or refresh requested; unloading model.")
            self.unload_model()

        self.initialized = True
        self.model_dir = model_dir
        self.device = requested_device
        self.torch_dtype = requested_dtype
        self.use_quantization = use_quantization
        self.use_flash_attention = use_flash_attention
        self.local_files_only = local_files_only

        self.model = None
        self.tokenizer = None
        self.generation_config = None

        self._load_tokenizer()
        self._load_model()
        self._load_generation_config()

        logger.info("DeepCoder module initialized successfully.")

    def unload_model(self):
        torch = get_torch()

        try:
            if getattr(self, "model", None) is not None:
                del self.model

            if getattr(self, "tokenizer", None) is not None:
                del self.tokenizer

            self.model = None
            self.tokenizer = None
            self.generation_config = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            self.initialized = False
            logger.info("DeepCoder unloaded successfully.")

        except Exception as e:
            logger.warning(f"DeepCoder unload encountered an issue: {repr(e)}")

    def _load_model(self):
        AutoModelForCausalLM = get_transformers("AutoModelForCausalLM")

        kwargs = {
            "torch_dtype": self.torch_dtype,
            "local_files_only": self.local_files_only,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }

        if self.device == "cuda":
            kwargs["device_map"] = "auto"

        if self.use_flash_attention:
            kwargs["attn_implementation"] = "flash_attention_2"

        if self.use_quantization and self.device == "cuda":
            try:
                BitsAndBytesConfig = get_transformers("BitsAndBytesConfig")
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except Exception as e:
                logger.warning(f"BitsAndBytesConfig unavailable; skipping quantization: {repr(e)}")
                self.use_quantization = False

        logger.info(f"Loading DeepCoder model from {self.model_dir}...")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            **kwargs,
        )

        if self.device != "cuda":
            self.model = self.model.to(self.device)

        self.model.eval()

        logger.info("DeepCoder model loaded and set to eval mode.")

    def _load_tokenizer(self):
        AutoTokenizer = get_transformers("AutoTokenizer")

        logger.info(f"Loading tokenizer from {self.model_dir}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
            use_fast=True,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        logger.info("Tokenizer loaded successfully.")

    def _load_generation_config(self):
        GenerationConfig = get_transformers("GenerationConfig")

        try:
            self.generation_config = GenerationConfig.from_pretrained(
                self.model_dir,
                local_files_only=self.local_files_only,
            )
        except Exception:
            self.generation_config = GenerationConfig()

        self.generation_config.do_sample = False
        self.generation_config.temperature = None
        self.generation_config.top_p = None
        self.generation_config.use_cache = True
        self.generation_config.max_new_tokens = min(
            int(getattr(self.generation_config, "max_new_tokens", 512) or 512),
            512,
        )

        logger.info("Generation config loaded.")

    def generate(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        use_chat_template: bool = False,
        messages: Optional[List[Dict[str, str]]] = None,
        do_sample: bool = False,
        return_full_text: bool = False,
        *args,
        **kwargs,
    ) -> str:
        torch = get_torch()

        if use_chat_template:
            final_messages = messages

            if final_messages is None and isinstance(prompt, list):
                final_messages = prompt

            if not final_messages:
                raise ValueError("use_chat_template=True requires messages or prompt as message list.")

            inputs = self.tokenizer.apply_chat_template(
                final_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            if hasattr(inputs, "to"):
                input_ids = inputs.to(self.device)
                model_inputs = {"input_ids": input_ids}
            else:
                model_inputs = {k: v.to(self.device) for k, v in inputs.items()}

        else:
            model_inputs = self.tokenizer(
                str(prompt),
                return_tensors="pt",
                padding=False,
                truncation=True,
            )
            model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        input_len = model_inputs["input_ids"].shape[-1]

        final_max = min(
            int(max_new_tokens or 256),
            int(getattr(self.generation_config, "max_new_tokens", 512) or 512),
        )

        generate_kwargs = {
            **model_inputs,
            "max_new_tokens": final_max,
            "do_sample": bool(do_sample),
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        with torch.inference_mode():
            outputs = self.model.generate(**generate_kwargs)

        if return_full_text:
            decoded_ids = outputs[0]
        else:
            decoded_ids = outputs[0][input_len:]

        return self.tokenizer.decode(
            decoded_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

    def get_model_info(self) -> Dict[str, Union[str, int, bool]]:
        return {
            "model_name": "DeepCoder-14B-Preview",
            "architecture": "Qwen2ForCausalLM",
            "num_layers": 48,
            "hidden_size": 5120,
            "vocab_size": 152064,
            "device": self.device,
            "torch_dtype": str(self.torch_dtype),
            "quantized": self.use_quantization,
            "flash_attention": self.use_flash_attention,
        }


_deep_coder: Optional[DeepCoder] = None


def get_deep_coder(
    module_path: Optional[str] = None,
    model_dir: Optional[str] = None,
    device: Optional[str] = None,
    torch_dtype=None,
    use_quantization: Optional[bool] = None,
    refresh_model: bool = False,
    use_flash_attention: bool = False,
    local_files_only: bool = True,
) -> DeepCoder:
    resolved_model_dir = model_dir or module_path or DEFAULT_PATH

    if use_quantization is None:
        use_quantization = False

    return DeepCoder(
        model_dir=resolved_model_dir,
        device=device,
        torch_dtype=torch_dtype,
        use_quantization=use_quantization,
        refresh_model=refresh_model,
        use_flash_attention=use_flash_attention,
        local_files_only=local_files_only,
    )


def _get_model(
    model_dir: Optional[str] = None,
    device: Optional[str] = None,
    torch_dtype=None,
    use_quantization: bool = False,
    refresh_model: bool = False,
    use_flash_attention: bool = False,
    local_files_only: bool = True,
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
            use_flash_attention=use_flash_attention,
            local_files_only=local_files_only,
        )
        return _deep_coder

    existing_config_changed = (
        getattr(_deep_coder, "model_dir", None) != resolved_model_dir
        or (
            device is not None
            and getattr(_deep_coder, "device", None) != device
        )
        or getattr(_deep_coder, "use_quantization", None) != use_quantization
        or getattr(_deep_coder, "use_flash_attention", None) != use_flash_attention
        or getattr(_deep_coder, "local_files_only", None) != local_files_only
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
            use_flash_attention=use_flash_attention,
            local_files_only=local_files_only,
        )

    return _deep_coder


def deep_coder_generate(
    prompt: Union[str, List[Dict[str, str]]],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    use_chat_template: bool = False,
    messages: Optional[List[Dict[str, str]]] = None,
    do_sample: bool = False,
    model_dir: Optional[str] = None,
    device: Optional[str] = None,
    torch_dtype=None,
    use_quantization: bool = False,
    refresh_model: bool = False,
    use_flash_attention: bool = False,
    local_files_only: bool = True,
    return_full_text: bool = False,
    *args,
    **kwargs,
) -> str:
    model = _get_model(
        model_dir=model_dir,
        device=device,
        torch_dtype=torch_dtype,
        use_quantization=use_quantization,
        refresh_model=refresh_model,
        use_flash_attention=use_flash_attention,
        local_files_only=local_files_only,
    )

    return model.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        use_chat_template=use_chat_template,
        messages=messages,
        do_sample=do_sample,
        return_full_text=return_full_text,
    )


def refresh_deep_coder(
    model_dir: Optional[str] = None,
    device: Optional[str] = None,
    torch_dtype=None,
    use_quantization: bool = False,
    use_flash_attention: bool = False,
) -> DeepCoder:
    return _get_model(
        model_dir=model_dir,
        device=device,
        torch_dtype=torch_dtype,
        use_quantization=use_quantization,
        refresh_model=True,
        use_flash_attention=use_flash_attention,
    )
