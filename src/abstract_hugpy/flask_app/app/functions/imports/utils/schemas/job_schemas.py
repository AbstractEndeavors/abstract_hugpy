from .imports import *
# jobs.py
@dataclass
class Job:
    id: str
    model_key: str
    status: JOBSTATUS = "queued"
    message: str = ""
    error: str | None = None
    progress: float = 0.0                 # 0.0–1.0
    total_bytes: int | None = None
    downloaded_bytes: int | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    # runtime-only: the download subprocess. Never serialized.
    _proc: "mp.Process | None" = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model_key": self.model_key,
            "status": self.status,
            "message": self.message,
            "error": self.error,
            "progress": round(self.progress, 4),
            "total_bytes": self.total_bytes,
            "downloaded_bytes": self.downloaded_bytes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, model_key: str) -> Job:
        job = Job(id=str(uuid.uuid4()), model_key=model_key)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def update(self, job_id: str, **changes) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            for k, v in changes.items():
                if hasattr(job, k):
                    setattr(job, k, v)
            job.updated_at = datetime.now(timezone.utc).isoformat()
            return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())


job_store = JobStore()
