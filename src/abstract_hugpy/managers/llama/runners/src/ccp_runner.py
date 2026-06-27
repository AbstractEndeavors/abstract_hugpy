from .base_runner import LlamaCppBaseRunner
from .imports import *
from ...serve import serve_endpoint, serve_model_name

def __init__(self, model_key, *, env_path=None):
    base = serve_endpoint(model_key)          # None for mode=off
    if not base:
        raise RuntimeError(f"{model_key}: no HTTP endpoint (mode=off) — use in-process")
    self.model_key = model_key
    self.base_url = base
    self.served_model = serve_model_name(model_key)
# ===========================================================================
# HTTP runner — talks to a running llama-server process
# ===========================================================================


class LlamaCppRunner(LlamaCppBaseRunner):
    def __init__(self, model_key: str, *, env_path: Optional[str] = None):

        base = serve_endpoint(model_key)          # None for mode=off
        if not base:
            raise RuntimeError(f"{model_key}: no HTTP endpoint (mode=off) — use in-process")
        self.model_key = model_key
        self.base_url = base
        self.served_model = serve_model_name(model_key)

##        cfg = load_llama_config(env_path=env_path)
##        self.model_key = model_key
##        self.llama_host: str = cfg["LLAMA_HOST"]
##        self.port: int = cfg[model_key]
##        self.base_url = f"{self.llama_host}:{self.port}"

    async def _iter_stream(self, messages, max_tokens, temp, top_p):
        payload = {"messages": messages, "max_tokens": max_tokens,
                   "temperature": temp, "top_p": top_p, "stream": True}
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", f"{self.base_url}/v1/chat/completions",
                                     json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or line.strip() == "[DONE]":
                        continue
                    line = line.removeprefix("data: ")
                    try:
                        data = json.loads(line)
                        choice = data["choices"][0]
                        text = (choice.get("delta") or {}).get("content") or ""
                        fr   = choice.get("finish_reason")
                    except Exception:
                        text, fr = "", None
                    yield text, fr
    def _chat_complete(self, messages, max_tokens, temp, top_p, stop):
        payload = {"messages": messages, "max_tokens": max_tokens,
                   "temperature": temp, "top_p": top_p, "stream": False}
        if stop:
            payload["stop"] = stop
        with httpx.Client(timeout=DEFAULT_HTTP_TIMEOUT) as client:
            r = client.post(f"{self.base_url}/v1/chat/completions", json=payload)
            r.raise_for_status()
            data = r.json()
        choice = data["choices"][0]
        return choice["message"]["content"] or "", choice.get("finish_reason") or "stop"

    def _raw_complete(self, prompt, max_tokens, temp, top_p, stop, return_full_text):
        payload = {"prompt": prompt, "n_predict": max_tokens,
                   "temperature": temp, "top_p": top_p, "stream": False}
        if stop:
            payload["stop"] = stop
        with httpx.Client(timeout=DEFAULT_HTTP_TIMEOUT) as client:
            r = client.post(f"{self.base_url}/completion", json=payload)
            r.raise_for_status()
            data = r.json()
        text = data.get("content") or data.get("text") or ""
        if return_full_text:
            text = prompt + text
        finish = data.get("stop_type") or data.get("finish_reason") or "stop"
        return text, finish
    def _blocking_complete(
        self,
        messages: list[dict] | str,
        max_tokens: int,
        temp: float,
        top_p: float,
        stop: Optional[list[str]],
        use_chat_template: bool,
        return_full_text: bool,
    ) -> tuple[str, str]:
        timeout = DEFAULT_HTTP_TIMEOUT

        if use_chat_template and isinstance(messages, list):
            payload = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temp,
                "top_p": top_p,
                "stream": False,
            }

            if stop:
                payload["stop"] = stop

            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
            finish = choice.get("finish_reason") or "stop"

            return text, finish

        # Raw-prompt fallback: llama-server /completion endpoint
        prompt = (
            messages
            if isinstance(messages, str)
            else messages_to_prompt_from_dicts(messages)
        )

        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temp,
            "top_p": top_p,
            "stream": False,
        }

        if stop:
            payload["stop"] = stop

        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{self.base_url}/completion",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        text = data.get("content") or data.get("text") or ""

        if return_full_text:
            text = prompt + text

        finish = data.get("stop_type") or data.get("finish_reason") or "stop"

        return text, finish
