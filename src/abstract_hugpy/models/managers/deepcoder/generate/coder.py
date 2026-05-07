# deepcoder/coder.py
import os.path as osp
from typing import Optional, List, Dict, Union
from .imports import get_torch, get_transformers, get_logFile, require, DEFAULT_PATHS
from .config import DeepCoderConfig
from typing import *
logger = get_logFile("deepcoder")


def _pick_device_and_dtype(torch, device: Optional[str], dtype) -> tuple[str, Any]:
    chosen_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is not None:
        return chosen_device, dtype
    if chosen_device == "cuda":
        return chosen_device, (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    if hasattr(torch.cpu, "is_bf16_supported") and torch.cpu.is_bf16_supported():
        return chosen_device, torch.bfloat16
    return chosen_device, torch.float32


def build_deepcoder_config(
    model_dir: Optional[str] = None,
    device: Optional[str] = None,
    torch_dtype=None,
    use_quantization: bool = False,
    use_flash_attention: bool = False,
    local_files_only: bool = True,
    max_new_tokens_cap: int = 512,
) -> DeepCoderConfig:
    torch = require("torch", reason="DeepCoder requires PyTorch")
    resolved_dir = model_dir or DEFAULT_PATHS.get("deepcoder")
    if not resolved_dir:
        raise ValueError("DeepCoder requires model_dir or DEFAULT_PATHS['deepcoder'].")
    if not osp.exists(resolved_dir):
        raise FileNotFoundError(f"DeepCoder model dir not found: {resolved_dir}")

    chosen_device, chosen_dtype = _pick_device_and_dtype(torch, device, torch_dtype)
    return DeepCoderConfig(
        model_dir=resolved_dir,
        device=chosen_device,
        torch_dtype=chosen_dtype,
        use_quantization=use_quantization and chosen_device == "cuda",
        use_flash_attention=use_flash_attention and chosen_device == "cuda",
        local_files_only=local_files_only,
        max_new_tokens_cap=max_new_tokens_cap,
    )


class DeepCoder:
    """Loaded model + tokenizer + generation_config. One instance per config."""

    def __init__(self, cfg: DeepCoderConfig):
        require("transformers", reason="DeepCoder requires HuggingFace transformers")
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self._load_tokenizer()
        self._load_model()
        self._load_generation_config()
        logger.info("DeepCoder ready: device=%s dtype=%s", cfg.device, cfg.torch_dtype)

    def _load_tokenizer(self):
        AutoTokenizer = get_transformers("AutoTokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_dir,
            trust_remote_code=True,
            local_files_only=self.cfg.local_files_only,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _load_model(self):
        AutoModelForCausalLM = get_transformers("AutoModelForCausalLM")
        kwargs = {
            "torch_dtype": self.cfg.torch_dtype,
            "local_files_only": self.cfg.local_files_only,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        if self.cfg.device == "cuda":
            kwargs["device_map"] = "auto"
        if self.cfg.use_flash_attention:
            kwargs["attn_implementation"] = "flash_attention_2"
        if self.cfg.use_quantization:
            BitsAndBytesConfig = get_transformers("BitsAndBytesConfig")
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.cfg.torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_dir, **kwargs)
        if self.cfg.device != "cuda":
            self.model = self.model.to(self.cfg.device)
        self.model.eval()

    def _load_generation_config(self):
        GenerationConfig = get_transformers("GenerationConfig")
        try:
            gc = GenerationConfig.from_pretrained(
                self.cfg.model_dir, local_files_only=self.cfg.local_files_only,
            )
        except Exception:
            gc = GenerationConfig()
        gc.do_sample = False
        gc.temperature = None
        gc.top_p = None
        gc.use_cache = True
        self.generation_config = gc

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
    ) -> str:
        torch = get_torch()

        # Boundary validation: reject overruns instead of silently capping.
        requested = int(max_new_tokens or 256)
        if requested > self.cfg.max_new_tokens_cap:
            raise ValueError(
                f"max_new_tokens={requested} exceeds cap={self.cfg.max_new_tokens_cap}; "
                f"raise the cap explicitly via DeepCoderConfig if intentional."
            )

        if use_chat_template:
            final_messages = messages or (prompt if isinstance(prompt, list) else None)
            if not final_messages:
                raise ValueError("use_chat_template=True requires messages or list-form prompt")
            inputs = self.tokenizer.apply_chat_template(
                final_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
            )
            model_inputs = (
                {"input_ids": inputs.to(self.cfg.device)} if hasattr(inputs, "to")
                else {k: v.to(self.cfg.device) for k, v in inputs.items()}
            )
        else:
            tok = self.tokenizer(str(prompt), return_tensors="pt", padding=False, truncation=True)
            model_inputs = {k: v.to(self.cfg.device) for k, v in tok.items()}

        input_len = model_inputs["input_ids"].shape[-1]

        gen_kwargs = {
            **model_inputs,
            "max_new_tokens": requested,
            "do_sample": bool(do_sample),
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.inference_mode():
            outputs = self.model.generate(**gen_kwargs)

        ids = outputs[0] if return_full_text else outputs[0][input_len:]
        return self.tokenizer.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=True,
        ).strip()


# ---- registry: one instance per unique config -----------------------------

_INSTANCES: dict[tuple, DeepCoder] = {}

def get_deep_coder(cfg: Optional[DeepCoderConfig] = None, **build_kwargs) -> DeepCoder:
    """Boot-time get. Pass a cfg or build one. Same key returns same instance."""
    if cfg is None:
        cfg = build_deepcoder_config(**build_kwargs)
    key = cfg.cache_key()
    if key not in _INSTANCES:
        _INSTANCES[key] = DeepCoder(cfg)
    return _INSTANCES[key]


def deep_coder_generate(prompt, **kwargs) -> str:
    """Convenience wrapper with the same surface as before."""
    gen_keys = {
        "max_new_tokens", "temperature", "top_p", "use_chat_template",
        "messages", "do_sample", "return_full_text",
    }
    gen_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in gen_keys}
    coder = get_deep_coder(**kwargs)  # remaining kwargs go to config builder
    return coder.generate(prompt=prompt, **gen_kwargs)
