import os
import sys
import importlib
from types import ModuleType
from functools import lru_cache
from typing import Optional, Tuple, Any, Dict
MODULE_DEFAULTS=[{
        "path": "/mnt/24T/hugging_face/modules/DeepCoder-14B",   # local folder (optional)
        "repo_id": "agentica-org/DeepCoder-14B-Preview",          # HF repo id (string!)
        "handle": "deepcoder"}
    ]

class ModelHubLoader:
    """
    All-in-one loader for:
      - robust/lazy imports (transformers, torch, keybert, whisper)
      - resolving sources from MODULE_DEFAULTS-style dicts
      - loading causal LM / seq2seq models from local dir or HF repo id

    Key features
    ------------
    • Avoids circular imports by sanitizing `sys.modules["transformers"]`
    • Prefers local directory if it exists; else uses repo id
    • Never passes dicts to `from_pretrained` (prevents HFValidationError)
    • Easy one-liners: load by module name from your defaults

    Example
    -------
    loader = ModelHubLoader(defaults=DEFAULT_PATHS)

    # Load DeepCoder (local if present, else repo id)
    tok, model = loader.load_causal_lm_by_name("deepcoder", device="cuda")

    # Or load explicitly from a source string (either local path or 'namespace/repo')
    tok, model = loader.load_causal_lm("agentica-org/DeepCoder-14B-Preview", device="cuda")
    """

    # ---------- construct ----------
    def __init__(self, defaults: Optional[Dict[str, Dict[str, str]]] = None):
        self._defaults = defaults or {}

    # ---------- defaults API ----------
    def set_defaults(self, defaults: Dict[str, Dict[str, str]]) -> None:
        """Replace the defaults mapping (MODULE_DEFAULTS / DEFAULT_PATHS shape)."""
        self._defaults = defaults or {}

    def get_default_record(self, name: str) -> Optional[Dict[str, str]]:
        """
        Return the defaults entry (with keys like: path, repo_type, handle).
        Example record:
          {
            "path": "/mnt/24T/hugging_face/modules/DeepCoder-14B",
            "repo_type": "agentica-org/DeepCoder-14B-Preview",
            "handle": "deepcoder"
          }
        """
        rec = self._defaults.get(name)
        if rec and not isinstance(rec, dict):
            raise TypeError(f"Defaults record for '{name}' must be a dict, got: {type(rec)}")
        return rec

    # ---------- source resolution ----------
    @staticmethod
    def _looks_like_local_dir(src: str) -> bool:
        return isinstance(src, str) and os.path.isdir(src)

    @staticmethod
    def _looks_like_repo_id(src: str) -> bool:
        # minimal sanity check for "namespace/repo"
        return isinstance(src, str) and ("/" in src) and (" " not in src) and not os.path.isdir(src)

    def resolve_src(
        self,
        name_or_src: str,
        prefer_local: bool = True,
        require_exists: bool = False,
        must_be_transformers_dir: bool = False,
    ) -> str:
        """
        Resolve a usable source string for .from_pretrained().
        Accepts either:
          • a module name found in defaults (uses its 'path' then 'repo_type'), or
          • a direct source string (local dir or repo id)

        Options:
          prefer_local: if defaults contain a 'path' and the path exists, use it first
          require_exists: if True and local path is chosen, enforce it exists
          must_be_transformers_dir: if True, require we see a config.json under the local dir
        """
        # If it's directly a usable string (path or repo id), return it
        if self._looks_like_local_dir(name_or_src) or self._looks_like_repo_id(name_or_src):
            return name_or_src

        # Otherwise treat as defaults key
        rec = self.get_default_record(name_or_src)
        if not rec:
            raise KeyError(
                f"No defaults entry for '{name_or_src}'. "
                "Pass a repo id like 'namespace/repo' or a local directory path, "
                "or add it to your defaults."
            )

        local_path = rec.get("path")
        repo_id = rec.get("repo_type")

        # prefer local
        if prefer_local and local_path and os.path.isdir(local_path):
            if must_be_transformers_dir and not os.path.isfile(os.path.join(local_path, "config.json")):
                # still usable by HF if it’s a valid Transformers folder; warn otherwise
                raise FileNotFoundError(
                    f"Local directory exists but does not look like a Transformers model dir: {local_path} "
                    "(missing config.json). Either export a proper model there or set must_be_transformers_dir=False."
                )
            return local_path

        # fall back to repo id
        if repo_id and self._looks_like_repo_id(repo_id):
            return repo_id

        # if they insisted the local must exist
        if require_exists and local_path:
            raise FileNotFoundError(f"Local model directory not found: {local_path}")

        # nothing valid
        raise ValueError(
            f"Defaults for '{name_or_src}' did not yield a valid source. "
            f"path={local_path!r}, repo_type={repo_id!r}"
        )

    # ---------- lazy/safe imports ----------
    @staticmethod
    @lru_cache(maxsize=None)
    def _lazy_import(module_name: str) -> ModuleType:
        mod = sys.modules.get(module_name)
        if mod is not None and getattr(mod, "__file__", None):
            return mod
        return importlib.import_module(module_name)

    @staticmethod
    def _safe_transformers() -> ModuleType:
        mod = sys.modules.get("transformers")
        if mod is not None:
            partial = getattr(mod, "__file__", None) is None or getattr(mod, "__spec__", None) is None
            if partial:
                # Remove partially initialized module to avoid "partially initialized" AttributeError
                sys.modules.pop("transformers", None)

        importlib.invalidate_caches()
        tf = importlib.import_module("transformers")

        # sanity check: catch local shadowing early
        tf_file = getattr(tf, "__file__", "") or ""
        if ("/site-packages/transformers/__init__.py" not in tf_file) and \
           ("/dist-packages/transformers/__init__.py" not in tf_file):
            raise RuntimeError(
                f"Unexpected transformers location: {tf_file}. "
                "You may have a local file named 'transformers.py' shadowing the HF library."
            )
        return tf

    # expose modules
    def transformers(self) -> ModuleType:
        return self._safe_transformers()

    def torch(self) -> ModuleType:
        return self._lazy_import("torch")

    def whisper(self) -> ModuleType:
        return self._lazy_import("whisper")

    def KeyBERT(self):
        return getattr(self._lazy_import("keybert"), "KeyBERT")

    # transformers symbols (getter methods to avoid top-level import)
    def AutoTokenizer(self):
        return getattr(self._safe_transformers(), "AutoTokenizer")

    def AutoModelForCausalLM(self):
        return getattr(self._safe_transformers(), "AutoModelForCausalLM")

    def AutoModelForSeq2SeqLM(self):
        return getattr(self._safe_transformers(), "AutoModelForSeq2SeqLM")

    def GenerationConfig(self):
        return getattr(self._safe_transformers(), "GenerationConfig")

    def pipeline(self):
        return getattr(self._safe_transformers(), "pipeline")

    # modeling_outputs
    def modeling_outputs(self) -> ModuleType:
        return self._lazy_import("transformers.modeling_outputs")

    def CausalLMOutputWithPast(self):
        return getattr(self.modeling_outputs(), "CausalLMOutputWithPast")

    # ---------- guarded .from_pretrained() ----------
    @staticmethod
    def _guard_src_for_from_pretrained(src: Any) -> str:
        """
        Ensure we pass a string (local path or repo id) into from_pretrained.
        This prevents the HFValidationError you saw when a dict was passed.
        """
        if isinstance(src, dict):
            raise TypeError(
                "from_pretrained() requires a string path or repo id, not a dict. "
                "Use ModelHubLoader.resolve_src(name) to get the actual string."
            )
        if not isinstance(src, str):
            raise TypeError(f"from_pretrained() expected a str path or repo id, got: {type(src)}")
        return src

    # ---------- high-level loaders (explicit src) ----------
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
        Load Causal LM + tokenizer from a *string* source (repo id or local dir).
        """
        src = self._guard_src_for_from_pretrained(src)
        AutoTokenizer = self.AutoTokenizer()
        AutoModelForCausalLM = self.AutoModelForCausalLM()

        tok = AutoTokenizer.from_pretrained(
            src, cache_dir=cache_dir, trust_remote_code=trust_remote_code
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
        Load Seq2Seq + tokenizer from a *string* source (repo id or local dir).
        """
        src = self._guard_src_for_from_pretrained(src)
        AutoTokenizer = self.AutoTokenizer()
        AutoModelForSeq2SeqLM = self.AutoModelForSeq2SeqLM()

        tok = AutoTokenizer.from_pretrained(
            src, cache_dir=cache_dir, trust_remote_code=trust_remote_code
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

    # ---------- convenience: load by module "name" using defaults ----------
    def load_causal_lm_by_name(
        self,
        name: str,
        *,
        prefer_local: bool = True,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        trust_remote_code: bool = True,
        torch_dtype: Optional[Any] = None,
        device_map: Optional[Any] = None,
        must_be_transformers_dir: bool = False,
        **kwargs,
    ) -> Tuple[Any, Any]:
        """
        Resolve src from defaults[name] (path vs repo) and load a causal LM.
        """
        src = self.resolve_src(
            name,
            prefer_local=prefer_local,
            require_exists=False,
            must_be_transformers_dir=must_be_transformers_dir,
        )
        return self.load_causal_lm(
            src,
            cache_dir=cache_dir,
            device=device,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            device_map=device_map,
            **kwargs,
        )

    def load_seq2seq_by_name(
        self,
        name: str,
        *,
        prefer_local: bool = True,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        trust_remote_code: bool = True,
        torch_dtype: Optional[Any] = None,
        device_map: Optional[Any] = None,
        must_be_transformers_dir: bool = False,
        **kwargs,
    ) -> Tuple[Any, Any]:
        """
        Resolve src from defaults[name] (path vs repo) and load a seq2seq model.
        """
        src = self.resolve_src(
            name,
            prefer_local=prefer_local,
            require_exists=False,
            must_be_transformers_dir=must_be_transformers_dir,
        )
        return self.load_seq2seq(
            src,
            cache_dir=cache_dir,
            device=device,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            device_map=device_map,
            **kwargs,
        )
