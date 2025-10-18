from .src import *
from typing import Optional, Dict, List
from abstract_utilities import SingletonMeta
def get_name(name=None):
    return name or "zerosearch"
def get_cache_dir(directory=None):
    return directory or "/mnt/24T/hugging_face/cache"
class DeepZeroManager(metaclass=SingletonMeta):
    """
    Singleton wrapper around GetModuleVars for the 'zerosearch' model.
    Mirrors DeepCoderManager: loader flags go to the manager/loader,
    generation flags go to .generate().
    """
    def __init__(
        self,
        *args,
        name: Optional[str] = None,            # default: "zerosearch"
        cache_dir: Optional[str] = None,       # optional HF cache dir
        trust_remote_code: bool = False,       # loader flag
        use_fast: bool = True,                 # tokenizer flag
        **kwargs                               # e.g., torch_dtype, device_map, use_quantization
    ):
        if getattr(self, "initialized", False):
            return

        self.initialized = True
        self.name = get_name(name)
        self.cache_dir = get_cache_dir(cache_dir)
        self.trust_remote_code = bool(trust_remote_code)
        self.use_fast = bool(use_fast)

        # Unified loader (tokenizer/model/config/device)
        # Must be able to resolve self.name via MODEL_SOURCES / MODEL_CACHE_DIRS.
        self.DeepZero = GetModuleVars(
            name=self.name,
            cache_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code,
            use_fast=self.use_fast,
            **kwargs
        )


def get_deepZeroManager(
    name: Optional[str] = None,
    *args,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = False,
    use_fast: bool = True,
    **kwargs
):
    return DeepZeroManager(
        name=name,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast,
        **kwargs
    )


def deep_zero_generate(
    prompt: str,
    *,
    name = None,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = True,
    use_fast: bool = True,
    max_new_tokens: int = 1000,
    temperature: float = 0.6,
    top_p: float = 0.95,
    use_chat_template: bool = False,
    messages: Optional[List[Dict[str, str]]] = None,
    do_sample: bool = False,
    **kwargs
) -> str:
    """
    Thin helper: obtains the singleton and calls its .generate().

    Only pass generation-related args here. Loader flags are consumed by the manager above.
    """
    mgr = get_deepZeroManager(
        name = get_name(name),
        cache_dir = get_cache_dir(cache_dir),
        trust_remote_code = bool(trust_remote_code),
        use_fast = bool(use_fast)
    )

    return mgr.DeepZero.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        use_chat_template=use_chat_template,
        messages=messages,
        do_sample=do_sample,
        **kwargs  # your GetModuleVars.generate will ignore unknown keys safely
    )


