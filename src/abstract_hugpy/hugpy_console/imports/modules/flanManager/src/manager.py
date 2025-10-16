from .imports import *
logger = logging.getLogger("FlanManager")

class FlanManager(BaseModelManager):
    """
    Managed interface for google/flan-t5-xl (text2text generation).
    Loads once per process, reuses shared TorchEnvManager context.
    """

    def __init__(self, model_dir=None, use_quantization=None):
        # BaseModelManager handles singleton setup and environment resolution
        super().__init__(modrl_name="flan", model_dir=model_dir, use_quantization=use_quantization)
        self.lock = threading.Lock()
    # ------------------------------------------------------------------
    # Model + tokenizer loading (overrides BaseModelManager)
    # ------------------------------------------------------------------
    def _load_model_and_tokenizer(self):
        AutoTokenizer = get_AutoTokenizer()
        AutoModelForSeq2SeqLM = get_AutoModelForSeq2SeqLM()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir).to(self.device)

        logger.info(f"Flan model loaded on {self.device} ({self.torch_dtype})")


    # ------------------------------------------------------------------
    # Custom summarization wrapper
    # ------------------------------------------------------------------
    def summarize(
        self,
        text: str,
        max_chunk: int = 512,
        min_length: int = 100,
        max_length: int = 512,
        do_sample: bool = False,
    ) -> str:
        if not self.pipeline:
            self._safe_preload()

        prompt = (
            "You are a highly observant assistant tasked with summarizing long, unscripted video monologues.\n\n"
            f"TEXT:\n{text}\n\n"
            "INSTRUCTIONS:\n"
            "Summarize the speaker’s core points and tone as if describing the monologue "
            "to someone who hasn’t heard it. Group related ideas together. Highlight interesting "
            "or unusual claims. Use descriptive language. Output a full narrative paragraph (or two), not bullet points."
        )

        result = self.pipeline(
            prompt,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            num_return_sequences=1,
        )[0]["generated_text"]

        return result.strip()



    def get_flan_summary(
        self,
        text: str,
        max_chunk: int = None,
        min_length: int = None,
        max_length: int = None,
        do_sample: bool = None
    ) -> str:
        prompt = f"""
You are a highly observant assistant tasked with summarizing long, unscripted video monologues.

TEXT:
{text}

INSTRUCTIONS:
Summarize the speaker’s core points and tone as if describing the monologue to someone who hasn’t heard it.
Group related ideas together. Highlight interesting or unusual claims. Use descriptive language.
Output a full narrative paragraph (or two), not bullet points.
"""
        max_chunk = zero_or_default(max_chunk,default=512)
        min_length = zero_or_default(min_length,default=100)
        max_length = zero_or_default(max_length,default=512)
        do_sample = do_sample or False
        return self.summarizer(prompt,
                          max_length=max_length,
                          min_length=min_length,
                          do_sample=do_sample)[0]['generated_text']
