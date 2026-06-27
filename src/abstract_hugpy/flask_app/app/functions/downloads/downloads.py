from .imports import *
# ---------------------------------------------------------------------------
# In-process job store
# ---------------------------------------------------------------------------

jobs: Dict[str, Dict] = {}
jobs_lock = threading.Lock()


def make_job(model_key: str) -> str:
    job_id = uuid.uuid4().hex[:10]
    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "model_key": model_key,
            "status": "queued",
            "message": "",
        }
    return job_id


def run_download(job_id: str, model_key: str) -> None:
    with jobs_lock:
        jobs[job_id]["status"] = "running"
    try:
        dest = download_model(model_key, MODELS[model_key])
        with jobs_lock:
            jobs[job_id]["status"] = "done"
            jobs[job_id]["message"] = str(dest)
    except Exception as exc:
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["message"] = str(exc)

