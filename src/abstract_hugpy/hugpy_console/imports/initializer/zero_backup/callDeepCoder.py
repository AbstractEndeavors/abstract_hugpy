from .src import *
from typing import *
from abstract_utilities import SingletonMeta
def connvert_to_args_kwargs(*args,**kwargs):
    return args,kwargs
class DeepCoderManager(metaclass=SingletonMeta):
    def __init__(self, *args,name=None,cache_dir: str = None,trust_remote_code: bool = False,use_fast=True,**kwargs):
        if not hasattr(self, "initialized"):
            self.name = name or "deepcoder"
            self.cache_dir = cache_dir or "/mnt/24T/hugging_face/cache"
            self.use_fast = True if use_fast is not False else use_fast
            self.trust_remote_code = True if trust_remote_code is not False else trust_remote_code
            self.DeepCoder = GetModuleVars(
                name=self.name,
                cache_dir=self.cache_dir,   # optional
                trust_remote_code=self.trust_remote_code,                    # if the repo needs it
                use_fast=self.use_fast,
                **kwargs
            )
            
def get_deepCoderManager(
     name=None,
     *args,
     cache_dir: str = None,
     trust_remote_code: bool = False,
     use_fast=True,
     **kwargs
     ):
     deepCoderMgr = DeepCoderManager(
         name=name,
         cache_dir=cache_dir,
         trust_remote_code=trust_remote_code,
         use_fast=use_fast,
         **kwargs
         )
     return deepCoderMgr

def deep_coder_generate(
    prompt: str,
    *,
    cache_dir=None,
    trust_remote_code=True,
    use_fast=True,
    max_new_tokens: int = 1000,
    temperature: float = 0.6,
    top_p: float = 0.95,
    use_chat_template: bool = False,
    messages: Optional[List[Dict[str, str]]] = None,
    do_sample: bool = False,
    **kwargs
):
    name = "deepcoder"
    mgr = get_deepCoderManager(
        name=name,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast,
    )

    # Only pass generation args to the model wrapper:
    return mgr.DeepCoder.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        use_chat_template=use_chat_template,
        messages=messages,
        do_sample=do_sample,
        **kwargs  # but your GetModuleVars.generate will filter these again
    )


