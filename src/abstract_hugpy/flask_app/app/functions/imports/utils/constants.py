from .imports import *
JOBSTATUS = Literal["queued", "running", "completed", "failed", "cancelled"]
hfApi = HfApi(token=HF_TOKEN)
GGUF_QUANT = re.compile(r"(Q\d+_[A-Z0-9_]+|F16|BF16|F32)", re.I)
