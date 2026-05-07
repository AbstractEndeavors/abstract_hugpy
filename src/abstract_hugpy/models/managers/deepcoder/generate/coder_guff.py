# deepcoder/coder_gguf.py — drop-in replacement, same surface
from llama_cpp import Llama

class DeepCoder:
    def __init__(self, cfg: DeepCoderConfig):
        self.cfg = cfg
        self.llm = Llama(
            model_path=cfg.model_dir,         # path to .gguf file now
            n_ctx=4096,
            n_threads=cfg.cpu_threads,        # add this field; default = physical cores
            n_gpu_layers=0,                   # CPU
            verbose=False,
        )

    def generate(self, prompt, max_new_tokens=256, temperature=0.0, top_p=1.0,
                 do_sample=False, **_):
        out = self.llm(
            prompt if isinstance(prompt, str) else self._format_messages(prompt),
            max_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.0,
            top_p=top_p,
            echo=False,
        )
        return out["choices"][0]["text"].strip()
