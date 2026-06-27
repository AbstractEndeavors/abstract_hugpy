import multiprocessing as mp
from datetime import datetime, timezone
from flask import jsonify, abort
from .imports import *
from .downloader import *
# ──────────────────────────────────────────────────────────────────────────
# Subprocess worker — module-level so it's spawn-safe; underscore-private so
# nothing imports it (multiprocessing references the function object directly).
# ──────────────────────────────────────────────────────────────────────────
def update_model_status(model: dict) -> dict:
    model.update(model_status(model))
    return model


def _estimate_total_bytes(model: dict) -> int | None:
    """Sum the sizes of exactly the files this download will fetch, so the
    progress bar can show a real percentage. Respects filename (single GGUF),
    include patterns, or full repo. Returns None on any failure -> the bar
    falls back to indeterminate, which still works."""
    hub_id = model.get("hub_id")
    if not hub_id:
        return None
    repo_id, _ = split_hub_id(hub_id)
    try:
        info = hfApi.model_info(repo_id, files_metadata=True)
    except Exception as exc:
        logger.info("size estimate failed for %s: %s", hub_id, exc)
        return None

    filename = model.get("filename")
    include = model.get("include")

    def will_download(path: str) -> bool:
        if filename:
            return path == filename or path.endswith("/" + filename)
        if include:
            pats = include if isinstance(include, list) else [include]
            return any(fnmatch.fnmatch(path, p) for p in pats)
        return True

    total = sum((s.size or 0) for s in (info.siblings or []) if will_download(s.rfilename))
    return total or None


def _download_worker(model_key: str, model: dict) -> None:
    os.setpgrp()
    download_one(model=model,model_key=model_key )   # writes hugpy.json via _stamp


def _dir_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total
# ──────────────────────────────────────────────────────────────────────────
# Launch: spawn the worker, then sample dir-size for progress in a monitor
# thread that also resolves the terminal state.
# ──────────────────────────────────────────────────────────────────────────
def start_cancellable_download(job: Job, model: dict, total_bytes: int | None = None) -> None:
    logger.info(model)
    dest = route_destination(model=model)
    logger.info(dest)
    proc = mp.Process(target=_download_worker, args=(job.model_key, model), daemon=True)
    proc.start()

    job_store.update(
        job.id, status="running", message="Downloading…",
        total_bytes=total_bytes, _proc=proc,
    )

    def monitor() -> None:
        nonlocal total_bytes
        # If the caller didn't know the size (registered-model downloads don't),
        # estimate it here so the bar can show a percentage. One HF metadata
        # call, on this background thread — never blocks the POST response.
        if total_bytes is None:
            total_bytes = _estimate_total_bytes(model)
            if total_bytes:
                job_store.update(job.id, total_bytes=total_bytes)

        while proc.is_alive():
            got = _dir_bytes(dest)
            pct = (got / total_bytes) if total_bytes else 0.0
            job_store.update(job.id, progress=min(pct, 0.999), downloaded_bytes=got)
            time.sleep(1.0)

        proc.join()

        cur = job_store.get(job.id)
        if cur and cur.status == "cancelled":
            return

        if proc.exitcode == 0:
            job_store.update(
                job.id, status="completed", progress=1.0,
                downloaded_bytes=_dir_bytes(dest),
                message=f"Installed at {dest}",
            )
            try:
                record_downloaded_model(model, dest)
                refresh_registry(run_discovery=False)
            except Exception as exc:
                logger.warning("post-download registry refresh failed: %s", exc)
        else:
            job_store.update(
                job.id, status="failed", message="Download failed.",
                error=f"worker exited with code {proc.exitcode}",
            )

    threading.Thread(target=monitor, daemon=True).start()



def cancel_download(job_id: str) -> dict:
    job = job_store.get(job_id)
    if not job:
        abort(404, description="Unknown job ID.")
    if job.status not in ("queued", "running"):
        return {"cancelled": False, "reason": f"job is {job.status}"}

    proc = getattr(job, "_proc", None)
    if proc is not None and proc.is_alive():
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

    job_store.update(job_id, status="cancelled", message="Cancelled by user.")
    return {"cancelled": True}
