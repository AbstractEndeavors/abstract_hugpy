from .imports import *
logger = logging.getLogger("BigBirdManager")


class BigBirdManager(BaseModelManager):
    """Persistent LED (Longformer-Encoder-Decoder) summarizer."""

    def _load_model_and_tokenizer(self):
        LEDTokenizer = get_LEDTokenizer()
        LEDForConditionalGeneration = get_LEDForConditionalGeneration()

        self.tokenizer = LEDTokenizer.from_pretrained(self.model_dir)
        self.model = LEDForConditionalGeneration.from_pretrained(self.model_dir).to(self.device)
        logger.info(f"BigBird model loaded on {self.device} ({self.torch_dtype})")
