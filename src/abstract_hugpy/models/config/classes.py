from __future__ import annotations
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import *
from huggingface_hub import snapshot_download
from abstract_security import get_env_value
from abstract_utilities import *
# ---------------------------------------------------------------------
# Model storage root
# ---------------------------------------------------------------------

MODEL_HOME = Path(
    get_env_value("MODEL_HOME")
    or os.path.expanduser("~/.cache/abstract_models")
)

def safe_dtype_name(value: Any) -> str:
    """
    Converts torch dtypes or dtype-like objects into stable string values.

    Examples:
        torch.float16  -> "torch.float16"
        torch.bfloat16 -> "torch.bfloat16"
        "auto"         -> "auto"
    """
    if value is None:
        return "None"

    return str(value)

# ---------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    name: str
    hub_id: str
    folder: str
    task: str
    framework: str = "transformers"
    filename: Optional[str] = None
    include: Optional[str] = None

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
    max_new_tokens_cap: int = 16000
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
