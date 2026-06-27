from .imports import *
class Runtime(str, Enum):
    transformers = "transformers"
    llama_cpp = "llama_cpp"
    dataset = "dataset"
    unknown = "unknown"


class FileSpec(BaseModel):
    path: str
    size: int | None = None

class ModelSpec(BaseModel):
    hub_id: str
    license: str | None = None
    gated: bool | str | None = None          # False | "auto" | "manual"
    last_modified: str | None = None
    total_bytes: int | None = None
    num_params: int | None = None             # from safetensors metadata
    context_length: int | None = None
    gguf_quants: list[str] = Field(default_factory=list)
    files: list[FileSpec] = Field(default_factory=list)




class ModelSearchResult(BaseModel):
    hub_id: str
    author: str | None = None
    downloads: int | None = None
    likes: int | None = None
    tags: list[str] = Field(default_factory=list)
    pipeline_tag: str | None = None
    library_name: str | None = None
    private: bool | None = None
    total_bytes: int | None = None
    last_modified: str | None = None
