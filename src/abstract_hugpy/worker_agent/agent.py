"""Standalone GPU worker agent for the abstract_hugpy LLM pool.

Run this on any box with a GPU and a working ``abstract_hugpy`` install to
donate that GPU's compute to the central console. The agent:

    1. Detects local GPUs.
    2. Registers with the central node (``/api/llm/workers/register``) and keeps
       a persistent worker id in a local state file so restarts reuse the row.
    3. Serves inference over HTTP for the models the central node assigns to it:
           GET  /health
           POST /infer          {model_key, messages|prompt, ...} -> {text, finish_reason}
           POST /infer/stream   -> SSE token/done/error events
       Inference runs through ``abstract_hugpy.managers.dispatch`` exactly like
       the central node, so the worker loads/serves the model on its own GPU.
    4. Heartbeats every ``--heartbeat`` seconds, reporting live GPU stats and
       which models are currently loaded.

The central node's chat route picks an online, assigned worker for the chosen
model and relays this agent's ``/infer/stream`` back to the browser. If no
worker is assigned (or all are offline) the central node runs the model
locally, so adding workers is purely additive.

Usage
-----
    python -m abstract_hugpy.worker_agent \
        --central https://abstractgpt.ai \
        --name gpu-box-1 \
        --host 10.0.0.5 --port 9100 \
        --models Qwen_Qwen2.5-7B-Instruct,meta-llama_Llama-3.1-8B-Instruct

Every flag also has an env fallback (WORKER_CENTRAL_URL, WORKER_NAME,
WORKER_HOST, WORKER_PORT, WORKER_MODELS, WORKER_ID_FILE, WORKER_HEARTBEAT).
"""
from __future__ import annotations

import os
import sys
import json
import time
import uuid
import socket
import logging
import argparse
import asyncio
import threading
import subprocess
import urllib.request
import urllib.error

from flask import Flask, request, jsonify, Response, stream_with_context

logger = logging.getLogger("abstract_hugpy.worker_agent")

# request_id -> asyncio.Event, so POST /infer/cancel can stop an in-flight
# stream mid-generation. Populated by _stream_sync, tripped by the cancel route.
_CANCELS: dict = {}


# ---------------------------------------------------------------------------
# GPU discovery
# ---------------------------------------------------------------------------
def detect_gpus() -> list[dict]:
    """Best-effort GPU inventory.

    Tries ``nvidia-smi`` first (no Python deps), then ``torch.cuda``. Returns
    an empty list on a CPU-only box — the worker still registers and serves,
    it just won't be fast.
    """
    gpus = _detect_gpus_nvidia_smi()
    if gpus:
        return gpus
    return _detect_gpus_torch()


def _detect_gpus_nvidia_smi() -> list[dict]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).decode("utf-8", "replace")
    except (OSError, subprocess.SubprocessError):
        return []

    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        idx, name, mem_total, mem_free = parts[:4]
        gpus.append(
            {
                "index": _safe_int(idx),
                "name": name,
                # nvidia-smi reports MiB; normalize to bytes.
                "memory_total": _safe_int(mem_total) * 1024 * 1024 if _safe_int(mem_total) else None,
                "memory_free": _safe_int(mem_free) * 1024 * 1024 if _safe_int(mem_free) else None,
            }
        )
    return gpus


def _detect_gpus_torch() -> list[dict]:
    try:
        import torch  # noqa: WPS433 (optional, lazy)

        if not torch.cuda.is_available():
            return []
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free, total = (None, getattr(props, "total_memory", None))
            try:
                free, total = torch.cuda.mem_get_info(i)
            except Exception:
                pass
            gpus.append(
                {
                    "index": i,
                    "name": props.name,
                    "memory_total": total,
                    "memory_free": free,
                }
            )
        return gpus
    except Exception:
        return []


def torch_cuda_status() -> dict:
    """Whether *torch* can actually use CUDA — distinct from nvidia-smi seeing a
    card. Inference runs on the GPU only when ``torch.cuda.is_available()`` is
    True; a CPU-only torch build (or a torch/CUDA-driver mismatch) leaves a
    perfectly good GPU unused. Surfaced in /health so this is diagnosable.
    """
    try:
        import torch
        available = bool(torch.cuda.is_available())
        return {
            "available": available,
            "device_count": torch.cuda.device_count() if available else 0,
            "device_name": torch.cuda.get_device_name(0) if available else None,
            "torch_version": getattr(torch, "__version__", None),
            "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def llama_cpp_cuda_status() -> dict:
    """Whether *llama.cpp* (GGUF backend) was built with GPU offload support.

    ``n_gpu_layers`` is silently ignored when llama-cpp-python is the CPU-only
    wheel, so a GGUF model runs entirely on CPU even though autofit picked GPU
    layers. ``llama_supports_gpu_offload()`` is the definitive build check.
    """
    try:
        import llama_cpp
        supports = None
        try:
            supports = bool(llama_cpp.llama_supports_gpu_offload())
        except Exception:
            pass
        return {
            "installed": True,
            "version": getattr(llama_cpp, "__version__", None),
            "supports_gpu_offload": supports,
        }
    except Exception as exc:  # noqa: BLE001
        return {"installed": False, "error": f"{type(exc).__name__}: {exc}"}


def _safe_int(value) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _local_ip_toward(central_url: str) -> str | None:
    """The worker's own LAN IP on the route it uses to reach central.

    Opening a UDP socket toward central (no packets are actually sent on
    connect) makes the kernel pick the source address it WOULD use — i.e. the
    worker's real outbound IP (e.g. 192.168.1.128), not loopback/127.0.1.1.

    This is what we advertise, because central can't derive it reliably: when
    the worker reaches central via a public domain, NAT hairpinning makes the
    source IP central sees the router's address (192.168.1.1), not the worker's.
    """
    from urllib.parse import urlparse
    try:
        parsed = urlparse(central_url)
        host = parsed.hostname or central_url
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(2.0)
            s.connect((host, port))
            ip = s.getsockname()[0]
        if ip and not ip.startswith("127."):
            return ip
    except OSError:
        pass
    return None


# ---------------------------------------------------------------------------
# Central node client (registration + heartbeat)
# ---------------------------------------------------------------------------
class CentralClient:
    def __init__(self, central_url: str):
        # Endpoints live under /api on the central Flask app.
        self.base = central_url.rstrip("/") + "/api/llm/workers"

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def register(self, payload: dict) -> dict:
        return self._post("/register", payload)

    def heartbeat(self, worker_id: str, payload: dict) -> dict:
        return self._post(f"/{worker_id}/heartbeat", payload)


# ---------------------------------------------------------------------------
# Local inference (reuses the same dispatch the central node uses)
# ---------------------------------------------------------------------------
def _ensure_present(payload: dict, central_url: str | None) -> None:
    """Provision the requested model before inference (central-first, HF fallback)."""
    model_key = payload.get("model_key")
    if not model_key:
        return
    try:
        from .provision import ensure_model_present, ensure_model_registered

        # Learn the model from central if the worker wasn't built with it, then
        # run inference against the canonical local key.
        canonical = ensure_model_registered(model_key, central_url)
        if canonical and canonical != model_key:
            payload["model_key"] = canonical
        ensure_model_present(payload.get("model_key"), central_url)
    except Exception as exc:
        logger.warning("provisioning check for %s failed: %s", model_key, exc)


def _ensure_present_streaming(payload: dict, central_url: str | None):
    """Provision the model, yielding SSE 'status' events with download progress.

    Yields encoded SSE lines (status/error). Returns normally once the model is
    present (or was already). Throttled so we don't flood the stream.
    """
    model_key = payload.get("model_key")
    if not model_key:
        return
    try:
        from .provision import (
            ensure_model_present, ensure_model_registered, model_is_local,
        )

        # Learn the model from central first, then work the rest of the stream
        # against the canonical local key (so resolution/loading can find it).
        canonical = ensure_model_registered(model_key, central_url)
        if canonical and canonical != model_key:
            payload["model_key"] = canonical
            model_key = canonical

        if model_is_local(model_key):
            return  # nothing to do; go straight to generation

        yield _sse({"type": "status", "stage": "provision",
                    "message": f"fetching {model_key}…", "progress": 0.0})

        # provision runs in a worker thread; it pushes (done,total,fname) onto a
        # queue that we drain into throttled SSE status events from this thread.
        import queue
        import threading

        q: "queue.Queue" = queue.Queue()
        result = {"ok": False, "err": None}

        def _progress(done, total, fname):
            q.put((done, total, fname))

        def _run():
            try:
                result["ok"] = ensure_model_present(model_key, central_url, progress=_progress)
            except Exception as exc:  # pragma: no cover
                result["err"] = exc
            finally:
                q.put(None)  # sentinel: done

        th = threading.Thread(target=_run, daemon=True)
        th.start()

        last_emit = 0.0
        while True:
            item = q.get()
            if item is None:
                break
            done, total, fname = item
            now = time.time()
            # Emit at most ~3x/sec, but always emit the first/last.
            if now - last_emit < 0.33 and done < (total or 1):
                continue
            last_emit = now
            frac = (done / total) if total else 0.0
            yield _sse({
                "type": "status", "stage": "provision",
                "message": f"downloading {model_key} ({_human(done)}/{_human(total)})",
                "progress": round(frac, 4),
                "done_bytes": done, "total_bytes": total, "file": fname,
            })
        th.join(timeout=1.0)

        if result["err"] is not None:
            yield _sse({"type": "error",
                        "message": f"provisioning failed: {result['err']}"})
            return
        if not result["ok"]:
            yield _sse({"type": "error",
                        "message": f"could not fetch model {model_key} from central or HF"})
            return
        yield _sse({"type": "status", "stage": "provision",
                    "message": "model ready, loading…", "progress": 1.0})
    except Exception as exc:
        logger.warning("streaming provisioning for %s failed: %s", model_key, exc)


def _human(n) -> str:
    if not n:
        return "?"
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(n)
    i = 0
    while v >= 1024 and i < len(units) - 1:
        v /= 1024
        i += 1
    return f"{v:.1f} {units[i]}"


def _materialize_file(payload: dict) -> str | None:
    """Rebuild an inlined upload (file_b64/file_name) into a local temp file.

    Central ships uploaded files as base64 since the worker can't see central's
    UPLOADS_HOME. We write the bytes to a temp file, point ``payload["file"]``
    at it, and return the temp path so the caller can delete it afterwards.
    Returns None when there's nothing to materialize.
    """
    b64 = payload.pop("file_b64", None)
    name = payload.pop("file_name", None)
    if not b64:
        return None
    import base64
    import tempfile

    suffix = ""
    if name and "." in name:
        suffix = "." + name.rsplit(".", 1)[-1]
    fd, tmp_path = tempfile.mkstemp(prefix="hugpy_worker_", suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(base64.b64decode(b64))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    payload["file"] = tmp_path
    return tmp_path


def _cleanup_file(path: str | None) -> None:
    if path:
        try:
            os.unlink(path)
        except OSError:
            pass


def _run_once(payload: dict) -> dict:
    from abstract_hugpy.managers.dispatch import execute_prompt

    tmp = _materialize_file(payload)
    try:
        result = execute_prompt(**payload)
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)

        if getattr(result, "ok", True):
            return {
                "ok": True,
                "text": getattr(result, "text", None) or str(result),
                "finish_reason": getattr(result, "finish_reason", None) or "stop",
            }
        return {"ok": False, "error": getattr(result, "error", None) or "run failed"}
    finally:
        _cleanup_file(tmp)


_SPILL_ENV = {
    "n_gpu_layers": "HUGPY_N_GPU_LAYERS",
    "gpu_mem_gib": "HUGPY_GPU_MEM_GIB",
    "cpu_mem_gib": "HUGPY_CPU_MEM_GIB",
    "tensor_split": "HUGPY_TENSOR_SPLIT",
    "main_gpu": "HUGPY_MAIN_GPU",
    "n_gpu": "HUGPY_N_GPU",
}


def _apply_spill(spill: dict | None) -> None:
    """Translate a per-request spill override dict into the env vars the spill
    module reads. Only set keys that were provided; the model loads lazily, so
    setting these before the first request for a model takes effect on load.

    NOTE: changing spill for an ALREADY-loaded model has no effect until it's
    evicted/reloaded — central can force that via a fresh worker process or by
    reassigning before first use. For the common case (assign, then chat) the
    override lands before the model is built.
    """
    if not spill:
        return
    for key, env_name in _SPILL_ENV.items():
        if key not in spill or spill[key] is None:
            continue
        val = spill[key]
        if isinstance(val, (list, tuple)):
            val = ",".join(str(x) for x in val)
        os.environ[env_name] = str(val)


def _sse(payload: dict) -> bytes:
    # werkzeug's WSGI server asserts the app yields bytes, not str — so encode
    # here. (gunicorn is more lenient, but the worker runs the dev server.)
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


# How many continuation passes we'll chain before giving up, so a runaway
# model can't loop forever. Each pass produces up to the per-call token cap.
_MAX_CONTINUATIONS = int(os.environ.get("WORKER_MAX_CONTINUATIONS", "20"))

# At a continuation seam, a model often re-emits the tail of the previous part.
# We look for an overlap up to this many characters and drop it.
_SEAM_WINDOW = int(os.environ.get("WORKER_SEAM_WINDOW", "400"))


def _overlap_len(prev_tail: str, seg: str) -> int:
    """Longest suffix of prev_tail that is also a prefix of seg.

    Used to strip a continuation seam where the model repeats text it already
    produced. Exact match (verbatim repetition is by far the common case).
    """
    maxk = min(len(prev_tail), len(seg))
    for k in range(maxk, 0, -1):
        if prev_tail.endswith(seg[:k]):
            return k
    return 0


def _run_one_pass(loop, payload: dict, cancel_event=None):
    """Run a single execute_prompt_stream pass.

    Yields ('token', text) tuples and finishes by setting the returned dict's
    'finish_reason' + accumulated 'text'. Generator returns the result dict.
    ``cancel_event`` lets the request be stopped mid-stream.
    """
    from abstract_hugpy.managers.dispatch import execute_prompt_stream

    agen = execute_prompt_stream(cancel_event=cancel_event, **payload)
    finish = "stop"
    try:
        while True:
            try:
                event = loop.run_until_complete(agen.__anext__())
            except StopAsyncIteration:
                break
            etype = getattr(event, "type", None)
            if etype == "token":
                yield ("token", getattr(event, "text", ""))
            elif etype == "done":
                finish = getattr(event, "finish_reason", None) or "stop"
                break
            elif etype == "error":
                yield ("error", getattr(event, "message", "run failed"))
                return {"finish_reason": "error"}
    except Exception as exc:
        # The runner raised instead of emitting an error event (e.g. a model
        # that needs infra this worker doesn't have). Convert to a clean error
        # event so the user sees a message, not a crashed stream / traceback.
        logger.warning("generation failed: %s: %s", type(exc).__name__, exc)
        yield ("error", f"{type(exc).__name__}: {exc}")
        return {"finish_reason": "error"}
    finally:
        try:
            loop.run_until_complete(agen.aclose())
        except Exception:
            pass
    return {"finish_reason": finish}


def _stream_sync(payload: dict, request_id: str | None = None):
    """Drive generation from Flask's sync ctx, with auto-continuation.

    When a pass stops because it hit the token cap (finish_reason == 'length' /
    'max_tokens'), we feed what was produced back as context and continue, so a
    response longer than any single token allowance still comes out complete.
    Emits 'status' events between continuation segments. The browser just keeps
    appending 'token' text, so continuation is seamless to the user.

    ``request_id`` registers an asyncio cancel Event so POST /infer/cancel can
    stop this stream mid-generation.
    """
    tmp = _materialize_file(payload)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Register a cancel Event for this request so /infer/cancel can trip it.
    cancel_event = asyncio.Event()
    if request_id:
        _CANCELS[request_id] = cancel_event

    # finish reasons that mean "ran out of room", i.e. continue.
    CONTINUE_ON = {"length", "max_tokens"}

    # Normalize to a messages list so we can append assistant partials.
    messages = payload.get("messages")
    if not messages:
        messages = [{"role": "user", "content": payload.get("prompt", "")}]
    base_kwargs = {k: v for k, v in payload.items() if k not in ("messages", "prompt")}

    try:
        full_text = ""
        for attempt in range(_MAX_CONTINUATIONS + 1):
            if cancel_event.is_set():
                yield _sse({"type": "done", "finish_reason": "cancelled"})
                return
            pass_kwargs = dict(base_kwargs)
            pass_kwargs["messages"] = messages
            if attempt > 0:
                yield _sse({"type": "status", "stage": "generate",
                            "message": f"continuing (part {attempt + 1})…",
                            "segment": attempt + 1})

            gen = _run_one_pass(loop, pass_kwargs, cancel_event=cancel_event)
            seg_text = ""        # raw text this pass produced (for the next prompt)
            errored = False

            # Seam dedup: on a continuation pass, buffer the head of the segment
            # until we have _SEAM_WINDOW chars (or the pass ends), strip any
            # overlap with what we already emitted, then stream the rest live.
            is_cont = attempt > 0
            prev_tail = full_text[-_SEAM_WINDOW:] if is_cont else ""
            buffering = is_cont
            head = ""

            def _emit(text):
                # helper so we both record full_text and yield the SSE token
                nonlocal full_text
                if not text:
                    return None
                full_text += text
                return _sse({"type": "token", "text": text})

            try:
                while True:
                    kind, data = next(gen)
                    if kind == "token":
                        seg_text += data
                        if buffering:
                            head += data
                            if len(head) < _SEAM_WINDOW:
                                continue
                            # enough buffered — drop the seam overlap, flush rest
                            k = _overlap_len(prev_tail, head)
                            ev = _emit(head[k:])
                            buffering = False
                            head = ""
                            if ev:
                                yield ev
                        else:
                            ev = _emit(data)
                            if ev:
                                yield ev
                    elif kind == "error":
                        yield _sse({"type": "error", "message": data})
                        errored = True
                        break
            except StopIteration as stop:
                result = stop.value or {"finish_reason": "stop"}

            # Pass ended while still buffering (short segment): flush remainder
            # minus the seam overlap.
            if not errored and buffering:
                k = _overlap_len(prev_tail, head)
                ev = _emit(head[k:])
                if ev:
                    yield ev
            if errored:
                return

            finish = result.get("finish_reason", "stop")
            if finish not in CONTINUE_ON:
                yield _sse({"type": "done", "finish_reason": finish})
                return
            if not seg_text.strip():
                # Hit the cap but produced nothing usable — stop to avoid a loop.
                yield _sse({"type": "done", "finish_reason": "stop"})
                return

            # Continue: append the partial assistant turn and prompt to keep going.
            messages = messages + [
                {"role": "assistant", "content": seg_text},
                {"role": "user", "content": "Continue exactly where you left off. "
                                            "Do not repeat any previous text."},
            ]

        # Exhausted the continuation budget.
        yield _sse({"type": "done", "finish_reason": "length"})
    except Exception as exc:
        # Last-resort guard: never let an exception escape into the WSGI layer
        # (that aborts the stream with a raw traceback). Emit a clean error.
        logger.warning("stream failed: %s: %s", type(exc).__name__, exc)
        yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
    finally:
        if request_id:
            _CANCELS.pop(request_id, None)
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        loop.close()
        _cleanup_file(tmp)


def loaded_model_keys() -> list[str]:
    try:
        from abstract_hugpy.managers.dispatch import loaded_model_keys as _loaded

        return sorted({mk for (mk, _task) in _loaded()})
    except Exception:
        return []


def _spill_describe() -> dict:
    try:
        from abstract_hugpy.managers.spill import describe

        return describe()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------
def build_app(state: "WorkerState") -> Flask:
    app = Flask("abstract_hugpy_worker")

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify(
            {
                "ok": True,
                "worker_id": state.worker_id,
                "name": state.name,
                "gpus": detect_gpus(),
                "cuda": torch_cuda_status(),
                "llama_cpp": llama_cpp_cuda_status(),
                "assigned_models": state.assigned_models,
                "provisioning": sorted(state._provisioning),
                "loaded_models": loaded_model_keys(),
                "spill": _spill_describe(),
            }
        )

    @app.route("/infer", methods=["POST"])
    def infer():
        payload = request.get_json(silent=True) or {}
        _apply_spill(payload.pop("spill", None))
        _ensure_present(payload, state.central_url)
        return jsonify(_run_once(payload))

    @app.route("/infer/stream", methods=["POST"])
    def infer_stream():
        payload = request.get_json(silent=True) or {}
        _apply_spill(payload.pop("spill", None))
        # Caller-supplied id for cancellation; else generate one. Echo it back
        # as the first SSE event so the client can cancel this exact request.
        req_id = str(payload.pop("request_id", "") or uuid.uuid4().hex)

        def _generate():
            yield _sse({"type": "request", "request_id": req_id})
            # Stream provisioning progress first (download from central/HF), then
            # generation with auto-continuation. Both emit SSE lines already.
            yield from _ensure_present_streaming(payload, state.central_url)
            yield from _stream_sync(payload, request_id=req_id)

        return Response(
            stream_with_context(_generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
            direct_passthrough=True,
        )

    @app.route("/infer/cancel/<request_id>", methods=["POST"])
    def infer_cancel(request_id):
        ev = _CANCELS.get(request_id)
        if ev is None:
            return jsonify({"cancelled": False, "reason": "unknown or finished request"}), 404
        ev.set()
        return jsonify({"cancelled": True, "request_id": request_id})

    @app.route("/probe/<path:model_key>", methods=["POST", "GET"])
    def probe(model_key):
        # Live VRAM-fit check: actually load the model on this worker's GPU and
        # report whether it fit, plus before/after free VRAM. Loading is cached
        # by dispatch, so a probe also warms the model for the first real chat.
        return jsonify(_probe_model(model_key, state))

    return app


def _free_vram_bytes() -> int | None:
    try:
        from abstract_hugpy.managers.spill import free_vram_bytes
        return free_vram_bytes()
    except Exception:
        return None


def _probe_model(model_key: str, state: "WorkerState") -> dict:
    """Load the model on the GPU and report fit + VRAM deltas.

    Returns {ok, fit, vram_free_before, vram_free_after, vram_used, error}.
    'fit' is a heuristic: ok load AND GPU memory actually decreased (i.e. weights
    landed on the GPU, not spilled entirely to CPU).
    """
    before = _free_vram_bytes()
    result: dict = {"model_key": model_key, "vram_free_before": before}
    try:
        # Learn the model from central (if needed), make sure its files are
        # present, then build the runner, which loads the model. A tiny run
        # confirms it can actually generate.
        from .provision import ensure_model_present, ensure_model_registered
        canonical = ensure_model_registered(model_key, state.central_url) or model_key
        ensure_model_present(canonical, state.central_url)

        from abstract_hugpy.managers.dispatch import runner_for
        runner_for(model_key=canonical)  # builds + caches the runner (loads weights)

        after = _free_vram_bytes()
        used = (before - after) if (before is not None and after is not None) else None
        result.update(
            ok=True,
            vram_free_after=after,
            vram_used=used,
            # If GPU free memory dropped meaningfully, weights are on the GPU.
            fit=bool(used and used > 64 * 1024 * 1024),
        )
    except Exception as exc:
        result.update(ok=False, fit=False, error=f"{type(exc).__name__}: {exc}")
    return result


# ---------------------------------------------------------------------------
# Agent lifecycle
# ---------------------------------------------------------------------------
class WorkerState:
    def __init__(self, name: str, url: str | None, worker_id: str | None,
                 central_url: str | None = None, port: int | None = None):
        self.name = name
        self.url = url            # None unless operator set --advertise/WORKER_URL
        self.worker_id = worker_id
        self.central_url = central_url
        self.port = port
        # Models central says we should serve, plus which we've already kicked
        # off a background provision for (so we don't re-trigger every beat).
        self.assigned_models: list[str] = []
        self._provisioning: set[str] = set()
        self._provision_lock = threading.Lock()


def _sync_assignment(state: "WorkerState", worker: dict) -> None:
    """React to central's worker record: adopt its model list and pre-provision.

    Central owns the assignment (set in the UI). The agent reads it back from
    every register/heartbeat response and, for any newly-assigned model it
    doesn't already have, downloads it in the background so the first chat
    doesn't pay the full download latency. Without this the worker never knew
    about UI allocation changes.
    """
    if not isinstance(worker, dict):
        return
    models = worker.get("models") or []
    if models == state.assigned_models:
        return
    state.assigned_models = list(models)
    logger.info("assignment updated: serving %s", models or "(nothing)")

    for model_key in models:
        with state._provision_lock:
            if model_key in state._provisioning:
                continue
            state._provisioning.add(model_key)

        def _bg(mk=model_key):
            try:
                from .provision import ensure_model_present, model_is_local
                if not model_is_local(mk):
                    logger.info("pre-provisioning assigned model %s…", mk)
                    ensure_model_present(mk, state.central_url)
                    logger.info("pre-provisioned %s", mk)
            except Exception as exc:
                logger.warning("pre-provision of %s failed: %s", mk, exc)
            finally:
                with state._provision_lock:
                    state._provisioning.discard(mk)

        threading.Thread(target=_bg, daemon=True).start()


def _load_worker_id(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return (json.load(fh) or {}).get("worker_id")
    except (OSError, ValueError):
        return None


def _save_worker_id(path: str, worker_id: str) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"worker_id": worker_id}, fh)
    except OSError:
        logger.warning("could not persist worker id to %s", path)


def _heartbeat_loop(client: CentralClient, state: WorkerState, args) -> None:
    while True:
        time.sleep(args.heartbeat)
        try:
            worker = client.heartbeat(
                state.worker_id,
                {
                    "gpus": detect_gpus(),
                    "loaded_models": loaded_model_keys(),
                    "spill": _spill_describe(),
                    "url": state.url,     # None -> central keeps source-IP URL
                    "port": state.port,
                },
            )
            # Adopt any assignment change made in the UI + pre-provision it.
            _sync_assignment(state, worker)
        except urllib.error.HTTPError as exc:
            if exc.code == 410:
                # Central forgot us (restart / cleared registry) — re-register.
                logger.warning("central returned 410; re-registering")
                _register(client, state, args)
            else:
                logger.warning("heartbeat HTTP %s", exc.code)
        except Exception as exc:
            logger.warning("heartbeat failed: %s", exc)


def _register(client: CentralClient, state: WorkerState, args) -> None:
    models = [m.strip() for m in (args.models or "").split(",") if m.strip()]
    payload = {
        "name": state.name,
        "url": state.url,            # None -> central uses the source IP
        "port": state.port,
        "gpus": detect_gpus(),
        "role": "worker",
        "models": models or None,
        "worker_id": state.worker_id,
    }
    worker = client.register(payload)
    state.worker_id = worker.get("id", state.worker_id)
    if state.worker_id:
        _save_worker_id(args.id_file, state.worker_id)
    # Adopt central's view of what we serve (it may already have assignments
    # for this worker_id from a previous session) and pre-provision them.
    _sync_assignment(state, worker)
    logger.info("registered as worker id=%s serving models=%s", state.worker_id, worker.get("models"))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="abstract_hugpy.worker_agent")
    p.add_argument("--central", default=os.environ.get("WORKER_CENTRAL_URL"),
                   help="Central console base URL, e.g. https://abstractgpt.ai")
    p.add_argument("--name", default=os.environ.get("WORKER_NAME", socket.gethostname()))
    p.add_argument("--host", default=os.environ.get("WORKER_HOST", "0.0.0.0"),
                   help="Bind address for the worker's inference server")
    p.add_argument("--port", type=int, default=int(os.environ.get("WORKER_PORT", "9100")))
    p.add_argument("--advertise", default=os.environ.get("WORKER_URL"),
                   help="URL the central node should call back on "
                        "(defaults to http://<host>:<port>)")
    p.add_argument("--models", default=os.environ.get("WORKER_MODELS", ""),
                   help="Comma-separated model_keys to self-assign on registration")
    p.add_argument("--heartbeat", type=float, default=float(os.environ.get("WORKER_HEARTBEAT", "15")))
    p.add_argument("--id-file", default=os.environ.get(
        "WORKER_ID_FILE", os.path.expanduser("~/.abstract_hugpy_worker.json")))

    # GPU/CPU spill defaults for this worker. These seed the spill env the
    # inference path reads; per-request overrides from central still win.
    spill = p.add_argument_group("spill (GPU/CPU split)")
    spill.add_argument("--spill", choices=["auto", "off"],
                       default=os.environ.get("WORKER_SPILL", "auto"),
                       help="auto = fit as many layers on GPU as VRAM allows "
                            "(spill rest to CPU); off = CPU only")
    spill.add_argument("--n-gpu-layers", type=int, default=_safe_int(os.environ.get("WORKER_N_GPU_LAYERS")),
                       help="llama.cpp: force N layers on GPU (overrides --spill)")
    spill.add_argument("--gpu-mem", type=float, default=_safe_float(os.environ.get("WORKER_GPU_MEM_GIB")),
                       help="transformers: per-GPU memory budget in GiB")
    spill.add_argument("--cpu-mem", type=float, default=_safe_float(os.environ.get("WORKER_CPU_MEM_GIB")),
                       help="transformers: CPU/RAM budget in GiB for offloaded layers")
    spill.add_argument("--tensor-split", default=os.environ.get("WORKER_TENSOR_SPLIT"),
                       help="multi-GPU split, comma-separated e.g. 0.7,0.3")
    spill.add_argument("--main-gpu", type=int, default=_safe_int(os.environ.get("WORKER_MAIN_GPU")),
                       help="primary GPU index")
    return p


def _safe_float(value) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _apply_cli_spill(args) -> None:
    """Seed the spill env from CLI flags (per-request overrides still win)."""
    if args.n_gpu_layers is not None:
        os.environ["HUGPY_N_GPU_LAYERS"] = str(args.n_gpu_layers)
    elif args.spill == "off":
        os.environ["HUGPY_N_GPU_LAYERS"] = "off"
    else:
        os.environ.setdefault("HUGPY_N_GPU_LAYERS", "auto")
    if args.gpu_mem is not None:
        os.environ["HUGPY_GPU_MEM_GIB"] = str(args.gpu_mem)
    if args.cpu_mem is not None:
        os.environ["HUGPY_CPU_MEM_GIB"] = str(args.cpu_mem)
    if args.tensor_split:
        os.environ["HUGPY_TENSOR_SPLIT"] = args.tensor_split
    if args.main_gpu is not None:
        os.environ["HUGPY_MAIN_GPU"] = str(args.main_gpu)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)

    if not args.central:
        print("error: --central (or WORKER_CENTRAL_URL) is required", file=sys.stderr)
        return 2

    _apply_cli_spill(args)

    # A worker runs vision models on its own GPU in-process; it has no separate
    # vision server to POST to. Force in-process unless the operator overrode it.
    os.environ.setdefault("HUGPY_VISION_INPROCESS", "1")

    # Only advertise a URL when the operator set one explicitly. Otherwise leave
    # it to central, which derives the reachable address from the request source
    # IP — far more reliable than the worker guessing past 127.0.1.1 / NAT / odd
    # NICs. We still send the listen port so central can build host:port.
    advertise = args.advertise
    if not advertise:
        # Determine the worker's own outbound IP on the route to central. This
        # is reliable even across NAT hairpinning, which fools central's
        # source-IP guess (central would see the router, e.g. 192.168.1.1, not
        # the worker's .128). Falls back to None -> central uses the source IP.
        ip = _local_ip_toward(args.central)
        if ip:
            advertise = f"http://{ip}:{args.port}"
            logger.info("advertising self as %s (local IP toward central)", advertise)
    # Surface GPU usability up front: a worker that can't use CUDA will silently
    # serve every model on CPU. Make that loud so it's not mistaken for "slow".
    _gpus = detect_gpus()
    _cuda = torch_cuda_status()
    _lcpp = llama_cpp_cuda_status()
    if _cuda.get("available"):
        logger.info("torch CUDA ready: %s (torch %s, cuda %s) — transformers models use the GPU",
                    _cuda.get("device_name"), _cuda.get("torch_version"),
                    _cuda.get("cuda_version"))
    elif _gpus:
        logger.warning(
            "GPU(s) detected by nvidia-smi (%s) but torch.cuda.is_available() is "
            "False — transformers inference will run on CPU. This worker's Python "
            "env needs a CUDA build of torch. torch=%s cuda=%s err=%s",
            ", ".join(g.get("name") or "?" for g in _gpus),
            _cuda.get("torch_version"), _cuda.get("cuda_version"), _cuda.get("error"))
    else:
        logger.warning("no usable GPU (nvidia-smi found none and torch has no CUDA); "
                       "inference will run on CPU")

    # GGUF models go through llama.cpp, which needs its OWN CUDA build.
    if _gpus and _lcpp.get("installed") and _lcpp.get("supports_gpu_offload") is False:
        logger.warning(
            "llama-cpp-python is installed WITHOUT GPU offload support — GGUF "
            "models will run on CPU regardless of n_gpu_layers. Reinstall with "
            "CUDA: CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install --force-reinstall "
            "--no-cache-dir llama-cpp-python  (llama_cpp %s)", _lcpp.get("version"))
    elif _gpus and _lcpp.get("supports_gpu_offload"):
        logger.info("llama.cpp GPU offload available (llama_cpp %s) — GGUF models "
                    "can use the GPU", _lcpp.get("version"))

    state = WorkerState(name=args.name, url=advertise,
                        worker_id=_load_worker_id(args.id_file),
                        central_url=args.central)
    state.port = args.port
    client = CentralClient(args.central)

    try:
        _register(client, state, args)
    except Exception as exc:
        logger.error("initial registration failed: %s", exc)
        # Keep going — the heartbeat loop will retry, and the server can still
        # serve a worker the operator registers manually.

    hb = threading.Thread(target=_heartbeat_loop, args=(client, state, args), daemon=True)
    hb.start()

    logger.info("worker inference server listening on %s (advertising %s)",
                f"{args.host}:{args.port}", state.url)
    build_app(state).run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
