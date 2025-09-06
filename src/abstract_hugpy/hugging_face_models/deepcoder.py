import torch
from abstract_utilities import SingletonMeta,get_logFile
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import BitsAndBytesConfig
from .config import DEFAULT_PATHS
from typing import *
DEFAULT_PATH = DEFAULT_PATHS["deepcoder"]
logger = get_logFile("deepcoder")
class DeepCoder(metaclass=SingletonMeta):
    """A robust Python module for interacting with the DeepCoder-14B-Preview model."""
    def __init__(
        self,
        model_dir: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: torch.dtype = torch.float16,
        use_quantization: bool = False
    ):
        """
        Initialize the DeepCoder module.

        Args:
            model_dir (str): Directory containing model weights and configuration files.
            device (str): Device to load the model on (e.g., 'cuda', 'cpu').
            torch_dtype (torch.dtype): Data type for model weights (e.g., torch.float16).
            use_quantization (bool): Whether to use 4-bit quantization for memory efficiency.
        """
        if not hasattr(self, 'initialized'):
            self.initialized=True
            self.model_dir = model_dir
            self.device = device
            self.torch_dtype = torch_dtype
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
        kwargs = {"torch_dtype": self.torch_dtype}
        if self.use_quantization and self.device == "cuda":
            try:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            except ImportError:
                self.use_quantization = False
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, **kwargs).to(self.device)

    def _load_tokenizer(self):
        """Load the tokenizer with the specified configuration."""
        logger.info(f"Loading tokenizer from {self.model_dir}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                logger.info("Set pad_token_id to eos_token_id.")
            logger.info("Tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {str(e)}")
            raise
    def _load_generation_config(self):
        """Load the generation configuration."""
        try:
            self.generation_config = GenerationConfig.from_pretrained(self.model_dir)
            logger.info("Generation configuration loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load generation config: {str(e)}")
            self.generation_config = GenerationConfig(
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                max_new_tokens=64000
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
        **kwargs
    ) -> str:
        """
        Generate text based on the input prompt or messages.

        Args:
            prompt (str): Input prompt for text generation.
            max_new_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature.
            top_p (float): Top-p sampling probability.
            use_chat_template (bool): Whether to apply the chat template.
            messages (List[Dict[str, str]]): List of messages for chat template.

        Returns:
            str: Generated text.
        """
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
                logger.warning("generation_config.max_new_tokens is not set or invalid. Using requested max_new_tokens only.")
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
                    eos_token_id=self.tokenizer.eos_token_id
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info("Text generation completed successfully.")
            return generated_text

        except Exception as e:
            logger.error(f"Text generation failed: {repr(e)}")
            raise


    def save_output(self, text: str, output_path: str):
        """
        Save generated text to a file.

        Args:
            text (str): Text to save.
            output_path (str): Path to save the text.
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
            logger.info(f"Output saved to {output_path}.")
        except Exception as e:
            logger.error(f"Failed to save output: {str(e)}")
            raise
    def get_model_info(self) -> Dict[str, Union[str, int]]:
        """
        Get information about the loaded model.

        Returns:
            Dict[str, Union[str, int]]: Model information.
        """
        return {
            "model_name": "DeepCoder-14B-Preview",
            "architecture": "Qwen2ForCausalLM",
            "num_layers": 48,
            "hidden_size": 5120,
            "vocab_size": 152064,
            "device": self.device,
            "torch_dtype": str(self.torch_dtype),
            "quantized": self.use_quantization
        }

def get_deep_coder(module_path=None,
                   torch_dtype=None,
                   use_quantization=None
                   ):
    module_path = module_path or DEFAULT_PATH
    torch_dtype = torch_dtype or torch.float16
    if use_quantization == None:
        use_quantization = True 
    deepcoder = DeepCoder(
        model_dir=module_path,
        torch_dtype=torch_dtype,
        use_quantization=use_quantization
        )
    return deepcoder
def try_deep_coder(module_path=None,
                   torch_dtype=None,
                   use_quantization=None):
    # Example usage
    try:
        # Initialize the DeepCoder module
        deepcoder = get_deep_coder(module_path=module_path,
                   torch_dtype=torch_dtype,
                   use_quantization=use_quantization)
        logger.info("DeepCoder logger initialized and active.")
        # Generate text from a prompt
        prompt = "Write a Python function to calculate the factorial of a number."
        generated_text = deepcoder.generate(
            prompt=prompt,
            max_new_tokens=2,
            use_chat_template=False
        )
        print("Generated Text:", generated_text)

        # Generate text using chat template
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Explain how to implement a binary search in Python."}
        ]
        chat_response = deepcoder.generate(
            prompt=messages,
            max_new_tokens=1000,
            use_chat_template=True
        )
        print("Chat Response:", chat_response)

        # Save output
        deepcoder.save_output(chat_response, "./output/binary_search_explanation.txt")

        # Get model info
        model_info = deepcoder.get_model_info()
        print("Model Info:", model_info)

    except Exception as e:
        logger.error(f"Example usage failed: {str(e)}")
