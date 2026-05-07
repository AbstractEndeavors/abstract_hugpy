# deepcoder/config.py
import os.path as osp
from dataclasses import dataclass, field
from typing import Optional, Any

@dataclass(frozen=True)
class DeepCoderConfig:
    model_dir: str
    device: str
    torch_dtype: Any                 # torch.dtype
    use_quantization: bool = False
    use_flash_attention: bool = False
    local_files_only: bool = True
    max_new_tokens_cap: int = 512    # hard ceiling, honored loudly

    def cache_key(self) -> tuple:
        """Hashable identity. Two configs with same key share an instance."""
        return (
            self.model_dir, self.device, str(self.torch_dtype),
            self.use_quantization, self.use_flash_attention, self.local_files_only,
        )
