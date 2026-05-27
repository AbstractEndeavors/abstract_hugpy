from .imports import *


class ModelConfig(BaseModel):
    """Registry entry for one model. Immutable after construction."""
    model_config = ConfigDict(frozen=True)

    name: str
    hub_id: str
    folder: str
    tasks: List[str]
    primary_task: str
    model_key: str
    framework: str = "transformers"
    filename: Optional[str] = None
    include: Optional[str] = None
    model_max_length: Optional[int] = DEFAULT_MAX_TOKENS
    port: Optional[int] = None
    host: Optional[str] = None
    timeout_s: Optional[int] = 3600

    def to_dict(self) -> dict:
        return self.model_dump()

    @model_validator(mode="after")
    def _check_primary_in_tasks(self) -> "ModelConfig":
        if self.primary_task not in self.tasks:
            raise ValueError(
                f"{self.model_key}: primary_task={self.primary_task!r} "
                f"not in tasks={sorted(self.tasks)!r}"
            )
        return self


@dataclass(frozen=True)
class DeepCoderRuntime:
    model_dir: str
    device: str
    torch_dtype: Any
    use_quantization: bool = False
    use_flash_attention: bool = False
    local_files_only: bool = True
    max_new_tokens_cap: int = DEFAULT_MAX_TOKENS
    max_concurrent_generations: int = 1

    def cache_key(self) -> tuple:
        return (
            self.model_dir,
            self.device,
            str(self.torch_dtype),
            self.use_quantization,
            self.use_flash_attention,
            self.local_files_only,
            self.max_new_tokens_cap,
            self.max_concurrent_generations,
        )

    def to_dict(self) -> dict:
        return {
            "model_dir": self.model_dir,
            "device": self.device,
            "torch_dtype": safe_dtype_name(self.torch_dtype),
            "use_quantization": self.use_quantization,
            "use_flash_attention": self.use_flash_attention,
            "local_files_only": self.local_files_only,
            "max_new_tokens_cap": self.max_new_tokens_cap,
            "max_concurrent_generations": self.max_concurrent_generations,
        }
