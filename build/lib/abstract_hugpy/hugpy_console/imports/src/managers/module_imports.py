import sys
import importlib
from types import ModuleType
from functools import lru_cache
from typing import Optional, Tuple, Any


class LibLoader:
    """
    One-stop lazy-and-safe import utility for:
      - Hugging Face transformers (with guards against circular/partial imports and local shadowing)
      - torch
      - keybert
      - whisper
    Also exposes convenience getters for common transformers classes and modeling_outputs,
    plus helper methods to load models safely.

    Usage:
        loader = LibLoader()

        # Get classes lazily (no top-level imports, avoids circulars)
        AutoTokenizer = loader.AutoTokenizer()
        AutoModelForCausalLM = loader.AutoModelForCausalLM()

        tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

        # Or use helpers:
        tok, model = loader.load_causal_lm("meta-llama/Llama-3.1-8B-Instruct", device="cuda")
    """

    # ------------------------------
    # Core lazy import primitives
    # ------------------------------
    @staticmethod
    @lru_cache(maxsize=None)
    def _lazy_import(module_name: str) -> ModuleType:
        """
        Import a module only once; return from sys.modules if already present and initialized.
        """
        mod = sys.modules.get(module_name)
        if mod is not None and getattr(mod, "__file__", None):
            return mod
        return importlib.import_module(module_name)

    # ------------------------------
    # Robust transformers import
    # ------------------------------
    @staticmethod
    def _safe_transformers() -> ModuleType:
        """
        Import the HF transformers package *robustly*:
          - If a partially-initialized module is in sys.modules (from circular import), drop it.
          - Invalidate caches before import.
          - Ensure we didn't import a locally shadowed 'transformers.py'.
        """
        mod = sys.modules.get("transformers")
        if mod is not None:
            partial = (
                getattr(mod, "__file__", None) is None
                or getattr(mod, "__spec__", None) is None
            )
            if partial:  # clear half-baked import
                sys.modules.pop("transformers", None)

        importlib.invalidate_caches()
        tf = importlib.import_module("transformers")

        tf_file = getattr(tf, "__file__", "") or ""
        # Basic sanity check: path should be site/dist-packages; adjust if your env differs
        if "/site-packages/transformers/__init__.py" not in tf_file and \
           "/dist-packages/transformers/__init__.py" not in tf_file:
            # Not strictly required, but helps catch local shadowing early
            raise RuntimeError(
                f"Unexpected transformers location: {tf_file}. "
                "A local module named 'transformers.py' may be shadowing the HF library."
            )
        return tf

    # Public accessor in case you want the module itself.
    def transformers(self) -> ModuleType:
        return self._safe_transformers()

    # ------------------------------
    # Direct getters for transformers symbols (callable to avoid top-level import)
    # ------------------------------
    def AutoModelForCausalLM(self):
        return getattr(self._safe_transformers(), "AutoModelForCausalLM")

    def AutoTokenizer(self):
        return getattr(self._safe_transformers(), "AutoTokenizer")

    def GenerationConfig(self):
        return getattr(self._safe_transformers(), "GenerationConfig")

    def pipeline(self):
        return getattr(self._safe_transformers(), "pipeline")

    def AutoModelForSeq2SeqLM(self):
        return getattr(self._safe_transformers(), "AutoModelForSeq2SeqLM")

    def T5TokenizerFast(self):
        return getattr(self._safe_transformers(), "T5TokenizerFast")

    def T5ForConditionalGeneration(self):
        return getattr(self._safe_transformers(), "T5ForConditionalGeneration")

    def LEDTokenizer(self):
        return getattr(self._safe_transformers(), "LEDTokenizer")

    def LEDForConditionalGeneration(self):
        return getattr(self._safe_transformers(), "LEDForConditionalGeneration")

    # ------------------------------
    # modeling_outputs accessors
    # ------------------------------
    def modeling_outputs(self) -> ModuleType:
        return self._lazy_import("transformers.modeling_outputs")

    def CausalLMOutputWithPast(self):
        return getattr(self.modeling_outputs(), "CausalLMOutputWithPast")

    # ------------------------------
    # Other libraries
    # ------------------------------
    def torch(self) -> ModuleType:
        return self._lazy_import("torch")

    def KeyBERT(self):
        return getattr(self._lazy_import("keybert"), "KeyBERT")

    def whisper(self) -> ModuleType:
        return self._lazy_import("whisper")

    # ------------------------------
    # High-level helpers for loading models safely
    # ------------------------------
    def load_causal_lm(
        self,
        src: str,
        *,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        trust_remote_code: bool = True,
        torch_dtype: Optional[Any] = None,
        device_map: Optional[Any] = None,
        **kwargs,
    ) -> Tuple[Any, Any]:
        """
        Load a Causal LM + tokenizer robustly.

        Args:
            src: HF repo id ('namespace/repo') or local directory path.
            cache_dir: Optional cache dir.
            device: e.g. 'cuda', 'cpu'. If None, leaves on default (usually CPU).
            trust_remote_code: passed through to HF loaders.
            torch_dtype: e.g. torch.float16; passed to model loader.
            device_map: passed to model loader (useful for accelerate/auto device map).
            **kwargs: forwarded to .from_pretrained()

        Returns:
            (tokenizer, model)
        """
        AutoTokenizer = self.AutoTokenizer()
        AutoModelForCausalLM = self.AutoModelForCausalLM()

        tok = AutoTokenizer.from_pretrained(
            src,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            src,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            device_map=device_map,
            **kwargs,
        )
        if device:
            model = model.to(device)
        return tok, model

    def load_seq2seq(
        self,
        src: str,
        *,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        trust_remote_code: bool = True,
        torch_dtype: Optional[Any] = None,
        device_map: Optional[Any] = None,
        **kwargs,
    ) -> Tuple[Any, Any]:
        """
        Load a Seq2Seq model + tokenizer (e.g., T5/LED) robustly.

        Args/Returns: same idea as load_causal_lm().
        """
        AutoTokenizer = self.AutoTokenizer()
        AutoModelForSeq2SeqLM = self.AutoModelForSeq2SeqLM()

        tok = AutoTokenizer.from_pretrained(
            src,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            src,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            device_map=device_map,
            **kwargs,
        )
        if device:
            model = model.to(device)
        return tok, model
