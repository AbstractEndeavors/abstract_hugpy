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

DEFAULT_PATH: str = DEFAULT_PATHS["deepcoder"]
logger = get_logFile("deepcoder")


class DeepCoder(metaclass=SingletonMeta):
    """A robust Python module for interacting with the DeepCoder-14B-Preview model."""

    def __init__(
        self,
        model_dir: str,
        device: Optional[str] = None,
        torch_dtype=None,
        use_quantization: bool = False,
    ):
        if hasattr(self, "initialized"):
            return

        # --- gate hard dependencies at init, not four calls later ---
        torch = require("torch", reason="DeepCoder requires PyTorch")
        require("transformers", reason="DeepCoder requires HuggingFace transformers")

        self.initialized = True
        self.model_dir = model_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or torch.float16
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
            raise

    def _load_model(self):
        AutoModelForCausalLM = get_transformers("AutoModelForCausalLM")

        kwargs = {"torch_dtype": self.torch_dtype}

        if self.use_quantization and self.device == "cuda":
            try:
                BitsAndBytesConfig = get_transformers("BitsAndBytesConfig")
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            except (ImportError, KeyError):
                logger.warning("BitsAndBytesConfig unavailable — skipping quantization.")
                self.use_quantization = False

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir, **kwargs
        ).to(self.device)

    def _load_tokenizer(self):
        AutoTokenizer = get_transformers("AutoTokenizer")

        logger.info(f"Loading tokenizer from {self.model_dir}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir, trust_remote_code=True
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                logger.info("Set pad_token_id to eos_token_id.")
            logger.info("Tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {str(e)}")
            raise

    def _load_generation_config(self):
        GenerationConfig = get_transformers("GenerationConfig")

        try:
            self.generation_config = GenerationConfig.from_pretrained(self.model_dir)
            logger.info("Generation configuration loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load generation config: {str(e)}")
            self.generation_config = GenerationConfig(
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                max_new_tokens=64000,
            )
            logger.info("Using default generation configuration.")

    def generate(
        self,
        prompt: str,
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

        try:
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

            config_max = self.generation_config.max_new_tokens
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
                )

            generated_text = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
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

    def get_model_info(self) -> Dict[str, Union[str, int]]:
        return {
            "model_name": "DeepCoder-14B-Preview",
            "architecture": "Qwen2ForCausalLM",
            "num_layers": 48,
            "hidden_size": 5120,
            "vocab_size": 152064,
            "device": self.device,
            "torch_dtype": str(self.torch_dtype),
            "quantized": self.use_quantization,
        }


def get_deep_coder(
    module_path: Optional[str] = None,
    torch_dtype=None,
    use_quantization: Optional[bool] = None,
) -> DeepCoder:
    module_path = module_path or DEFAULT_PATH
    if torch_dtype is None:
        torch_dtype = get_torch().float16
    if use_quantization is None:
        use_quantization = True
    return DeepCoder(
        model_dir=module_path,
        torch_dtype=torch_dtype,
        use_quantization=use_quantization,
    )


def try_deep_coder(
    module_path: Optional[str] = None,
    torch_dtype=None,
    use_quantization: Optional[bool] = None,
):
    try:
        deepcoder = get_deep_coder(
            module_path=module_path,
            torch_dtype=torch_dtype,
            use_quantization=use_quantization,
        )
        logger.info("DeepCoder logger initialized and active.")

        prompt = "Write a Python function to calculate the factorial of a number."
        generated_text = deepcoder.generate(
            prompt=prompt, max_new_tokens=2, use_chat_template=False
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
            prompt=messages, max_new_tokens=1000, use_chat_template=True
        )
        print("Chat Response:", chat_response)

        deepcoder.save_output(chat_response, "./output/binary_search_explanation.txt")
        print("Model Info:", deepcoder.get_model_info())

    except Exception as e:
        logger.error(f"Example usage failed: {str(e)}")
