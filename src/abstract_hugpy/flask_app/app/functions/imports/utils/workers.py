"""GPU worker registry.

A *worker* is a remote box that runs the standalone worker agent
(``abstract_hugpy.worker_agent``), exposes an HTTP inference endpoint, and
joins this central node so its GPU(s) can serve one or more models from the
manifest.

This module is the single source of truth for the pool. It owns:

    - persistence of the worker list to a JSON file beside the model manifest
      (so the pool survives restarts),
    - registration / heartbeat / removal,
    - model assignment (which worker may serve which model_key),
    - liveness (a worker is ``online`` only if it has heartbeat-ed recently),
    - selection (pick an online worker that is assigned + ready for a model).

Routing (chat/streaming) and the ``/llm/workers`` routes are dumb consumers of
the functions exported here.
"""
from __future__ import annotations

import os
import json
import time
import uuid
import threading
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

try:
    import fcntl  # POSIX advisory file locks — cross-process coordination.
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

from .schemas import settings


def _default_workers_path() -> str:
    """Sit the worker registry next to the model manifest (…/projects/)."""
    return os.path.join(os.path.dirname(settings.manifest_path), "workers.json")


# A worker that hasn't checked in within this window is considered offline.
HEARTBEAT_TIMEOUT_SECONDS = 45.0


def _now() -> float:
    return time.time()


def _is_online(worker: Dict[str, Any]) -> bool:
    last = worker.get("last_seen") or 0
    return (_now() - last) <= HEARTBEAT_TIMEOUT_SECONDS


def _public_view(worker: Dict[str, Any]) -> Dict[str, Any]:
    """The shape returned to API callers — derived ``status`` included."""
    return {**worker, "status": "online" if _is_online(worker) else "offline"}


def _match_keys(model_key: str) -> set:
    """Normalized aliases a model might be named by, for tolerant matching.

    A model can be referenced as its registry key, its hub_id (owner/name), or
    just the trailing name — and with different case. We compare on the set of
    these forms so an assignment made via one spelling still routes a chat that
    uses another. Example: "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
    "Qwen2.5-Coder-3B-Instruct-GGUF" and the lowercased variants all match.
    """
    if not model_key:
        return set()
    raw = str(model_key).strip()
    forms = {raw, raw.lower()}
    tail = raw.split("/")[-1]
    forms.add(tail)
    forms.add(tail.lower())
    return forms


class WorkerStore:
    """Disk-authoritative, multi-process-safe registry of GPU workers.

    Under gunicorn/uwsgi the API runs as several processes, so an in-memory
    dict would split-brain: a worker registered in process A would be invisible
    to a heartbeat or chat request handled by process B (the classic symptom is
    "registers + shows in the UI, but heartbeats 410 and chats never offload").

    To avoid that, ``workers.json`` is the single source of truth: every read
    re-loads it, and every mutation takes an exclusive ``fcntl`` lock, reloads,
    mutates, and writes back atomically. A short-lived in-process RLock just
    keeps threads within one process from racing the same fd.
    """

    # Read-cache TTL: the console polls /llm/workers every ~10s; without this
    # every poll does an open+flock+read of workers.json, which BLOCKS on a
    # degraded mount and stalls the API. Reads serve from cache within the TTL;
    # writes always go to disk and refresh the cache, so liveness stays correct.
    _READ_TTL = 3.0

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = path or _default_workers_path()
        self._lock = threading.RLock()
        self._cache: Optional[Dict[str, Dict[str, Any]]] = None
        self._cache_at = 0.0
        self._ensure_parent()

    # -- persistence (disk-authoritative) ----------------------------------
    def _ensure_parent(self) -> None:
        parent = os.path.dirname(self._path)
        if parent:
            try:
                os.makedirs(parent, exist_ok=True)
            except OSError:
                pass

    def _read_unlocked(self, fh=None) -> Dict[str, Dict[str, Any]]:
        """Parse the workers map from an open fh, or from disk if none given."""
        try:
            if fh is not None:
                fh.seek(0)
                raw = fh.read()
            elif os.path.exists(self._path):
                with open(self._path, "r", encoding="utf-8") as f:
                    raw = f.read()
            else:
                return {}
            if not raw.strip():
                return {}
            data = json.loads(raw)
            if isinstance(data, dict):
                return {w["id"]: w for w in data.get("workers", []) if w.get("id")}
        except (OSError, ValueError, KeyError):
            return {}
        return {}

    def _write_unlocked(self, fh, workers: Dict[str, Dict[str, Any]]) -> None:
        """Overwrite the open, locked fh with the workers map."""
        payload = json.dumps({"workers": list(workers.values())}, indent=2)
        fh.seek(0)
        fh.truncate()
        fh.write(payload)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass

    def _load(self) -> Dict[str, Dict[str, Any]]:
        """Read-only snapshot of the registry, cached for a few seconds.

        Polls (list/get/pick) hit this; the cache keeps a hung/slow mount from
        blocking every request. Writes refresh the cache, so freshly-registered
        or reassigned workers are visible immediately to the writing process.
        """
        now = time.time()
        with self._lock:
            if self._cache is not None and (now - self._cache_at) < self._READ_TTL:
                return self._cache
            data = self._read_unlocked()
            self._cache = data
            self._cache_at = now
            return data

    @contextmanager
    def _transaction(self):
        """Yield the on-disk workers map under an exclusive cross-process lock.

        Reload -> mutate (caller) -> persist. The yielded dict is written back
        when the block exits without raising. Falls back to a plain in-process
        critical section when ``fcntl`` is unavailable.
        """
        with self._lock:
            self._ensure_parent()
            # Open r+ (create if missing) so we hold one fd for lock+read+write.
            fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o644)
            fh = os.fdopen(fd, "r+", encoding="utf-8")
            try:
                if fcntl is not None:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                workers = self._read_unlocked(fh)
                yield workers
                self._write_unlocked(fh, workers)
                # Refresh the read-cache so this process sees its own write
                # immediately (and other processes within the TTL).
                self._cache = workers
                self._cache_at = time.time()
            finally:
                try:
                    if fcntl is not None:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                finally:
                    fh.close()

    # -- registration / lifecycle ------------------------------------------
    def register(
        self,
        *,
        name: str,
        url: str,
        gpus: Optional[List[Dict[str, Any]]] = None,
        role: str = "worker",
        models: Optional[List[str]] = None,
        worker_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add a worker (or re-register an existing one by id/url).

        Re-registration is keyed first on the supplied ``worker_id``, then on
        ``url`` — so an agent that restarts and advertises the same URL keeps
        its assignments instead of creating a duplicate row.
        """
        url = (url or "").rstrip("/")
        with self._transaction() as workers:
            existing = None
            if worker_id and worker_id in workers:
                existing = workers[worker_id]
            else:
                for w in workers.values():
                    if w.get("url") == url:
                        existing = w
                        break

            if existing is not None:
                existing.update(
                    name=name or existing.get("name"),
                    url=url or existing.get("url"),
                    gpus=gpus if gpus is not None else existing.get("gpus", []),
                    role=role or existing.get("role", "worker"),
                    last_seen=_now(),
                )
                if models is not None:
                    existing["models"] = sorted(set(models))
                return _public_view(existing)

            wid = worker_id or uuid.uuid4().hex
            worker = {
                "id": wid,
                "name": name or wid,
                "url": url,
                "role": role or "worker",
                "gpus": gpus or [],
                "models": sorted(set(models or [])),
                "created_at": _now(),
                "last_seen": _now(),
            }
            workers[wid] = worker
            return _public_view(worker)

    def heartbeat(
        self,
        worker_id: str,
        *,
        gpus: Optional[List[Dict[str, Any]]] = None,
        loaded_models: Optional[List[str]] = None,
        spill: Optional[Dict[str, Any]] = None,
        url: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Mark a worker alive and refresh its live GPU / loaded-model stats."""
        with self._transaction() as workers:
            worker = workers.get(worker_id)
            if worker is None:
                return None
            worker["last_seen"] = _now()
            if url:
                worker["url"] = url.rstrip("/")
            if gpus is not None:
                worker["gpus"] = gpus
            if loaded_models is not None:
                worker["loaded_models"] = loaded_models
            if spill is not None:
                worker["spill"] = spill
            return _public_view(worker)

    def remove(self, worker_id: str) -> bool:
        with self._transaction() as workers:
            return workers.pop(worker_id, None) is not None

    # -- model assignment ---------------------------------------------------
    def assign_model(
        self,
        worker_id: str,
        model_key: str,
        spill: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Assign a model to a worker, with optional per-assignment spill config.

        ``spill`` is an opaque dict of GPU/CPU knobs (e.g. n_gpu_layers,
        gpu_mem_gib, cpu_mem_gib) the worker applies when it loads the model.
        Omitted / None means "use the worker's autofit default."
        """
        with self._transaction() as workers:
            worker = workers.get(worker_id)
            if worker is None:
                return None
            models = set(worker.get("models", []))
            models.add(model_key)
            worker["models"] = sorted(models)
            if spill is not None:
                by_model = worker.setdefault("spill_by_model", {})
                # An empty dict clears any override back to autofit.
                if spill:
                    by_model[model_key] = spill
                else:
                    by_model.pop(model_key, None)
            return _public_view(worker)

    def unassign_model(self, worker_id: str, model_key: str) -> Optional[Dict[str, Any]]:
        with self._transaction() as workers:
            worker = workers.get(worker_id)
            if worker is None:
                return None
            worker["models"] = sorted(set(worker.get("models", [])) - {model_key})
            worker.get("spill_by_model", {}).pop(model_key, None)
            return _public_view(worker)

    def spill_for(self, worker_id: str, model_key: str) -> Dict[str, Any]:
        """Per-assignment spill override for (worker, model), or {} for autofit."""
        worker = self._load().get(worker_id)
        if worker is None:
            return {}
        return dict(worker.get("spill_by_model", {}).get(model_key, {}))

    # -- queries ------------------------------------------------------------
    def get(self, worker_id: str) -> Optional[Dict[str, Any]]:
        worker = self._load().get(worker_id)
        return _public_view(worker) if worker else None

    def all(self) -> List[Dict[str, Any]]:
        return [_public_view(w) for w in self._load().values()]

    def workers_for_model(self, model_key: str, *, online_only: bool = True) -> List[Dict[str, Any]]:
        wanted = _match_keys(model_key)
        out = []
        for w in self.all():
            assigned = w.get("models", [])
            # Match on the raw key OR any normalized alias (hub_id vs key vs
            # case), so an assignment made via one form still routes a chat that
            # names the model a slightly different way.
            if not (model_key in assigned or wanted & {a for m in assigned for a in _match_keys(m)}):
                continue
            if online_only and w["status"] != "online":
                continue
            out.append(w)
        return out

    def pick_for_model(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Choose an online worker to serve ``model_key``.

        Preference order:
            1. workers that already report the model as loaded (warm),
            2. otherwise the least-recently-picked online assignee.

        Returns ``None`` when no online worker is assigned to the model, which
        signals the caller to fall back to local execution.
        """
        candidates = self.workers_for_model(model_key, online_only=True)
        if not candidates:
            # Fall back to assigned workers even with a stale heartbeat. Heartbeat
            # (worker->central) can time out when central is briefly slow, while
            # offload (central->worker) still works — so an assigned worker that
            # looks "offline" is often still serviceable. The stream proxy fails
            # fast to local if the worker is genuinely unreachable.
            candidates = self.workers_for_model(model_key, online_only=False)
        if not candidates:
            return None

        warm = [w for w in candidates if model_key in (w.get("loaded_models") or [])]
        pool = warm or candidates
        # Spread load: prefer the assignee touched longest ago.
        pool.sort(key=lambda w: w.get("last_picked", 0))
        chosen = pool[0]

        # Persist the pick so round-robin survives across processes.
        with self._transaction() as workers:
            stored = workers.get(chosen["id"])
            if stored is not None:
                stored["last_picked"] = _now()
                chosen = stored
        return _public_view(chosen)


worker_store = WorkerStore()


# Module-level convenience wrappers (mirrors the manifest.py / peers.py style of
# exposing plain functions for routes to import).
def register_worker(**kwargs) -> Dict[str, Any]:
    return worker_store.register(**kwargs)


def heartbeat_worker(worker_id: str, **kwargs) -> Optional[Dict[str, Any]]:
    # kwargs: gpus, loaded_models, spill — all optional, passed straight through.
    return worker_store.heartbeat(worker_id, **kwargs)


def remove_worker(worker_id: str) -> bool:
    return worker_store.remove(worker_id)


def assign_model(worker_id: str, model_key: str,
                 spill: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    return worker_store.assign_model(worker_id, model_key, spill=spill)


def unassign_model(worker_id: str, model_key: str) -> Optional[Dict[str, Any]]:
    return worker_store.unassign_model(worker_id, model_key)


def spill_for(worker_id: str, model_key: str) -> Dict[str, Any]:
    return worker_store.spill_for(worker_id, model_key)


def list_workers() -> List[Dict[str, Any]]:
    return worker_store.all()


def get_worker(worker_id: str) -> Optional[Dict[str, Any]]:
    return worker_store.get(worker_id)


def pick_worker_for_model(model_key: str) -> Optional[Dict[str, Any]]:
    return worker_store.pick_for_model(model_key)
