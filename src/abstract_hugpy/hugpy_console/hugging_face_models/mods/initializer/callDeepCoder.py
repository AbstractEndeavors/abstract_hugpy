from .src import *
def connvert_to_args_kwargs(*args,**kwargs):
    return args,kwargs
def deep_coder_generate(
        prompt: str,
        cache_dir = None,
        trust_remote_code=True,
        use_fast=True,
        max_new_tokens: int = 1000,
        temperature: float = 0.6,
        top_p: float = 0.95,
        use_chat_template: bool = False,
        messages: Optional[List[Dict[str, str]]] = None,
        do_sample: bool = False,
        *args,
        **kwargs
    ):
    name = "deepcoder"
    cache_dir = cache_dir or "/mnt/24T/hugging_face/cache"
    args,kwargs = connvert_to_args_kwargs(prompt=prompt,
            cache_dir = cache_dir,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            use_chat_template=use_chat_template,
            messages=messages,
            do_sample=do_sample,
            *args,
            **kwargs)
    dc = GetModuleVars(
        name=name,
        cache_dir=cache_dir,   # optional
        trust_remote_code=trust_remote_code,                    # if the repo needs it
        use_fast=use_fast
    )

    return dc.generate(*args,**kwargs)
