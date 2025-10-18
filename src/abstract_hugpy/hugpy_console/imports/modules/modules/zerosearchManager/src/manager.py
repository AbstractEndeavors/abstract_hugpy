from ..imports import *
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union
import threading


logger = get_logFile("zerosearch")


def _resolve_zerosearch_source():
    """
    Decide what to pass to from_pretrained(...):
      - Prefer a local folder with config.json/weights if it exists
      - Else use a proper repo id (namespace/repo_name)
    """
    z = DEFAULT_PATHS.get("zerosearch") or {}
    local_path = (z or {}).get("path")
    repo_id    = (z or {}).get("repo_id")  # previously called repo_type in your mapping

    # sanity: pick a model source, not a dataset
    if local_path and os.path.isdir(local_path):
        return local_path
    return repo_id  # must be a *model* repo id, not a dataset id


class ZeroSearch(BaseModelManager):
    """Persistent ZeroSearch LLM interface optimized for long-running inference."""

    def __init__(self, model_dir: Optional[str] = None, use_quantization: bool = False, **kwargs):
        if hasattr(self, "initialized") and self.initialized:
            return

        self.initialized = True
        self.lock = threading.Lock()

        # Torch env (your helper)
        env = TorchEnvManager()
        self.torch = env.torch
        self.device = env.device
        self.dtype = env.dtype
        self.use_quantization = use_quantization or env.use_quantization

        # Decide model source
        chosen = model_dir or _resolve_zerosearch_source()
        if isinstance(chosen, dict):
            # guard against accidental dict here
            chosen = chosen.get("path") or chosen.get("repo_id")
        self.model_dir = chosen

        # Components
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.generation_config = None

        logger.info(
            f"ZeroSearch initializing on {self.device} ({self.dtype}) [quantized={self.use_quantization}]"
        )
        self._preload_async()

    # ---------------- preload ----------------
    def _preload_async(self):
        thread = threading.Thread(target=self._safe_preload, daemon=True)
        thread.start()

    def _safe_preload(self):
        try:
            self._load_model_and_tokenizer()
            logger.info("ZeroSearch model preloaded successfully.")
        except Exception as e:
            logger.exception(f"ZeroSearch preload failed: {e}")

    # ---------------- loading ----------------
    def _load_model_and_tokenizer(self):
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_model = ex.submit(self._load_model)
            f_tok   = ex.submit(self._load_tokenizer)
            self.model = f_model.result()
            self.tokenizer = f_tok.result()

        self._load_generation_config()
        self._create_pipeline()

    def _load_model(self):
        if not self.model_dir:
            raise RuntimeError(
                "ZeroSearch: no model_dir/repo_id configured. "
                "If this is a dataset, you must point to a real *model* repo."
            )

        logger.info(f"Loading ZeroSearch model from {self.model_dir}...")
        AutoModelForCausalLM = get_AutoModelForCausalLM()

        kwargs = {"torch_dtype": self.dtype}
        if "cuda" in self.device:
            kwargs["device_map"] = "auto"

        if self.use_quantization and "cuda" in self.device:
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
                logger.info("Using 4-bit quantization.")
            except Exception:
                logger.warning("bitsandbytes not available; skipping quantization.")

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
                do_sample=True, temperature=0.6, top_p=0.95, max_new_tokens=1024
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

    # ---------------- generation ----------------
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.95,
        use_chat_template: bool = False,
        messages: Optional[List[Dict[str, str]]] = None,
        do_sample: bool = True,
        **kwargs,
    ) -> str:
        """
        Thread-safe generation. Only pass *generation* kwargs to model.generate.
        """
        with self.lock:
            # lazy load if needed
            if self.model is None or self.tokenizer is None:
                logger.info("Lazy-loading ZeroSearch model on first request...")
                self._safe_preload()
                # Wait until loaded in this thread (simple guard)
                if self.model is None or self.tokenizer is None:
                    raise RuntimeError("ZeroSearch model not loaded yet.")

            # 1) Build inputs (handle chat template BEFORE generate)
            if use_chat_template and messages:
                if hasattr(self.tokenizer, "apply_chat_template"):
                    inputs = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    )
                else:
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

            # 2) Keep only generation kwargs
            allowed_gen_keys = {
                "max_new_tokens","min_new_tokens",
                "temperature","top_p","top_k",
                "num_beams","length_penalty",
                "repetition_penalty","no_repeat_ngram_size",
                "do_sample","early_stopping",
                "eos_token_id","pad_token_id","bos_token_id",
                "num_return_sequences","return_dict_in_generate",
                "output_scores","use_cache",
            }
            gen_kwargs = {k: v for k, v in kwargs.items() if k in allowed_gen_keys}
            gen_kwargs.setdefault("max_new_tokens", max_new_tokens)
            gen_kwargs.setdefault("temperature", temperature)
            gen_kwargs.setdefault("top_p", top_p)
            gen_kwargs.setdefault("do_sample", do_sample)
            gen_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
            gen_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)

            # deterministic path: drop sampling-only knobs
            if not gen_kwargs.get("do_sample", False):
                gen_kwargs.pop("temperature", None)
                gen_kwargs.pop("top_p", None)

            try:
                with self.torch.no_grad():
                    outputs = self.model.generate(**inputs, **gen_kwargs)
                text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return text.strip()

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.error("OOM detected; retrying on CPU...")
                    self._recover_to_cpu()
                    return self.generate(
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        use_chat_template=use_chat_template,
                        messages=messages,
                        do_sample=do_sample,
                        **kwargs,
                    )
                raise

    # ---------------- recovery & info ----------------
    def _recover_to_cpu(self):
        try:
            if self.device.startswith("cuda"):
                self.torch.cuda.empty_cache()
        except Exception:
            pass
        self.device = "cpu"
        if self.model is not None:
            self.model.to("cpu")
        logger.warning("ZeroSearch model moved to CPU due to GPU memory constraints.")

    def get_info(self) -> Dict[str, Union[str, int]]:
        return {
            "model_name": "ZeroSearch",
            "model_dir": str(self.model_dir),
            "device": self.device,
            "dtype": str(self.dtype),
            "quantized": bool(self.use_quantization),
            "initialized": self.model is not None,
        }
