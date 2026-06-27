from .imports import *
from .specs_schemas import *
class DownloadStatus(str, Enum):
    queued = "queued"
    running = "running"
    complete = "complete"
    failed = "failed"




class DownloadRequest(BaseModel):
    hub_id: str = Field(..., examples=["Qwen/Qwen2.5-Coder-3B-Instruct-GGUF"])
    framework: Runtime | str = Field(default="transformers")
    task: str = Field(default="text-generation")
    filename: str | None = None
    include: str | list[str] | None = None
    repo_type: Literal["model", "dataset"] = "model"


class DownloadJob(BaseModel):
    job_id: str
    hub_id: str
    framework: str
    task: str
    destination: str
    status: DownloadStatus
    error: str | None = None
