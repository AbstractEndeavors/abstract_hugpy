import threading
import uuid
import traceback

# job states: queued → running → done/failed
jobs = {}

def start_job(func, *args, **kwargs):
    """Start a job in a background thread, return job_id."""
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "result": None, "error": None}

    def run():
        jobs[job_id]["status"] = "running"
        try:
            result = func(*args, **kwargs)
            jobs[job_id]["status"] = "done"
            jobs[job_id]["result"] = result
        except Exception as e:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = f"{e}\n{traceback.format_exc()}"

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return job_id


def get_job(job_id):
    return jobs.get(job_id, {"status": "not_found"})
