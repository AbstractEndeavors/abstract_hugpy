## Torch Environment Manager (singleton shared by all managers)
from ..imports import *
from ..module_imports import *
logger = logging.getLogger("TorchEnvManager")
class TorchEnvManager(metaclass=SingletonMeta):
    """
    Centralized device and torch environment manager.
    Determines device, dtype, and quantization flags once per process.
    """

    def __init__(self):
        if not hasattr(self, "initialized"):
            self.initialized = True
          

 
            self.torch = get_torch()
            self.device = self._determine_device()
            self.dtype = self._determine_dtype()
            self.gpu_count = self._count_gpus()
            self.use_quantization = self._should_quantize()

            logger.info(
                f"TorchEnvManager initialized: device={self.device}, dtype={self.dtype}, "
                f"quantized={self.use_quantization}, gpus={self.gpu_count}"
            )

    # ------------------------------------------------------------------
    # Internal logic
    # ------------------------------------------------------------------
    def _determine_device(self) -> str:
        """Pick CUDA device if available, else CPU."""
        if self.torch.cuda.is_available():
            # Optional: allow manual override via env var
            gpu_index = os.getenv("MODEL_GPU", "0")
            return f"cuda:{gpu_index}"
        return "cpu"

    def _determine_dtype(self):
        """Choose optimal torch dtype for current device."""
        return self.torch.float16 if "cuda" in self._determine_device() else self.torch.float32

    def _count_gpus(self) -> int:
        try:
            return self.torch.cuda.device_count()
        except Exception:
            return 0

    def _should_quantize(self) -> bool:
        """Decide whether to enable 4-bit quantization."""
        quant_env = os.getenv("USE_QUANTIZATION", "true").lower()
        return quant_env in ("1", "true", "yes")

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
    def get_env(self):
        """Return a dictionary of all torch environment details."""
        return {
            "torch": self.torch,
            "device": self.device,
            "dtype": self.dtype,
            "gpu_count": self.gpu_count,
            "quantized": self.use_quantization,
        }

    def summary(self) -> str:
        return (
            f"TorchEnv(device={self.device}, dtype={self.dtype}, "
            f"quantized={self.use_quantization}, gpus={self.gpu_count})"
        )
def get_torch_env_manager():
    torch_env_mgr = TorchEnvManager()
    return torch_env_mgr
def get_torch_mgr_torch():
    torch_env_mgr = TorchEnvManager()
    return torch_env_mgr.torch
def get_torch_mgr_device():
    torch_env_mgr = TorchEnvManager()
    return torch_env_mgr.device
def get_torch_mgr_dtype():
    torch_env_mgr = TorchEnvManager()
    return torch_env_mgr.dtype
def get_torch_mgr_gpu_count():
    torch_env_mgr = TorchEnvManager()
    return torch_env_mgr.gpu_count
def get_torch_mgr_use_quantization():
    torch_env_mgr = TorchEnvManager()
    return torch_env_mgr.use_quantization
def get_torch_mgr_env():
    torch_env_mgr = TorchEnvManager()
    return torch_env_mgr.get_env
def get_torch_mgr_summary():
    torch_env_mgr = TorchEnvManager()
    return torch_env_mgr.summary
