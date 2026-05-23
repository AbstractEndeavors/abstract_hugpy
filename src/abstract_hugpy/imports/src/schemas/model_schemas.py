from .imports import *
@dataclass(frozen=True)
class ModelConfig:
    name: str
    hub_id: str
    folder: str
    tasks: list
    primary_task: str
    model_key:str
    framework: str = "transformers"
    filename: Optional[str] = None
    include: Optional[str] = None
    model_max_length: Optional[str] = DEFAULT_MAX_TOKENS
    port: Optional[int] = None
    host: Optional[int] = None
    timeout_s: Optional[int] = 3600
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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

    def to_dict(self) -> dict[str, Any]:
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
