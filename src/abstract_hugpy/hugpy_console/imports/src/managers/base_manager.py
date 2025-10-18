## Base Model Manager Template
from ..imports import *
from ..module_imports import *
from .torchEnvManager import *
logger = logging.getLogger(__name__)

class BaseModelManager(metaclass=SingletonMeta):
    """Generic lazy-loaded manager for HuggingFace or SentenceTransformer models."""

    def __init__(self,modrl_name=None , model_dir=None, use_quantization=False):
        self.model_dir = model_dir or DEFAULT_PATHS.get(modrl_name)
        self.use_quantization = use_quantization or get_torch_mgr_use_quantization()
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = get_torch_mgr_device()
        self.torch_dtype = get_torch_mgr_dtype()
        self.lock = threading.Lock()
        self.initialized = False

    def _resolve_env(self):
        torch = get_torch_mgr_torch()
        self.device = get_torch_mgr_device()
        self.torch_dtype = get_torch_mgr_dtype()
        return torch

    def preload(self):
        """Asynchronous preload in background."""
        if self.initialized:
            return
        self.initialized = True
        thread = threading.Thread(target=self._safe_preload, daemon=True)
        thread.start()

    def _safe_preload(self):
        try:
            self._load_model_and_tokenizer()
            self._create_pipeline()
            logger.info(f"{self.__class__.__name__} initialized successfully.")
        except Exception as e:
            logger.error(f"{self.__class__.__name__} preload failed: {e}")

    def _load_model_and_tokenizer(self):
        raise NotImplementedError("Subclasses must implement model/tokenizer loading")

    def _create_pipeline(self):
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model and tokenizer must be loaded first.")
        self.pipeline = get_pipeline()(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
        )

    def generate(self, prompt, **kwargs):
        """Thread-safe generation wrapper."""
        with self.lock:
            if not self.pipeline:
                self._safe_preload()
            return self.pipeline(prompt, **kwargs)[0]["generated_text"]
class BigBirdManager(BaseModelManager):
    def _load_model_and_tokenizer(self):
        LEDTokenizer = get_LEDTokenizer()
        LEDForConditionalGeneration = get_LEDForConditionalGeneration()
        self.tokenizer = LEDTokenizer.from_pretrained(self.model_dir)
        self.model = LEDForConditionalGeneration.from_pretrained(self.model_dir)
class FlanManager(BaseModelManager):
    def _load_model_and_tokenizer(self):
        AutoTokenizer = get_AutoTokenizer()
        AutoModelForSeq2SeqLM = get_AutoModelForSeq2SeqLM()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir)

        
