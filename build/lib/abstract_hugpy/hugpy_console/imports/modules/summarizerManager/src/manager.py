from ..imports import *

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
SUMMARIZER_DEFAULT_DIR ="/mnt/24T/hugging_face/modules/text_summarization/"
MODEL_NAME = "gpt-4"
CHUNK_OVERLAP = 30
DEFAULT_CHUNK_TOK = 450
SHORTCUT_THRESHOLD = 200

# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    return text

def clean_output_text(text: str) -> str:
    text = re.sub(r'["]{2,}', '"', text)
    text = re.sub(r'\.{3,}', '...', text)
    text = re.sub(r"[^\w\s\.,;:?!\-'\"()]+", "", text)
    return text.strip()

def chunk_text(text: str, max_tokens: int) -> List[str]:
    return recursive_chunk(
        text=text,
        desired_tokens=max_tokens,
        model_name=MODEL_NAME,
        separators=["\n\n", "\n", r"(?<=[\.?!])\s", ", ", " "],
        overlap=CHUNK_OVERLAP,
    )

def scale_lengths(mode: str, tokens: int) -> Tuple[int, int]:
    m = mode.lower()
    if m == "short":
        return max(16, int(tokens * 0.1)), max(40, int(tokens * 0.25))
    if m == "medium":
        return max(32, int(tokens * 0.25)), max(80, int(tokens * 0.5))
    if m == "long":
        return max(64, int(tokens * 0.35)), max(150, int(tokens * 0.7))
    return max(32, int(tokens * 0.2)), max(120, int(tokens * 0.6))

# ------------------------------------------------------------------
# Manager
# ------------------------------------------------------------------
class SummarizerManager(metaclass=SingletonMeta):
    """Lazy-loaded T5 summarizer manager."""

    def __init__(self):
        if not hasattr(self, "initialized"):
            self.initialized = True
            self.lock = threading.Lock()
            # Lazy-load tokenizer & model once
            self.tokenizer = get_T5TokenizerFast().from_pretrained(SUMMARIZER_DEFAULT_DIR)
            self.model = get_T5ForConditionalGeneration().from_pretrained(SUMMARIZER_DEFAULT_DIR)
            cfg_path = os.path.join(SUMMARIZER_DEFAULT_DIR, "generation_config.json")
            cfg = safe_read_from_json(cfg_path)
            self.gen_cfg = get_GenerationConfig()(**cfg)
    def summarize_chunk(self, text: str, min_length: int, max_length: int) -> str:
        inputs = self.tokenizer(
            "summarize: " + normalize_text(text),
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        gen_cfg = self.gen_cfg
        gen_cfg.min_length = min_length
        gen_cfg.max_length = max_length
        gen_cfg.no_repeat_ngram_size = 3
        gen_cfg.num_beams = 4
        gen_cfg.early_stopping = True

        torch = get_torch()
        with torch.no_grad():
            out = self.model.generate(inputs.input_ids, **gen_cfg.to_dict())
        return clean_output_text(self.tokenizer.decode(out[0], skip_special_tokens=True))

# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def get_summarizer_summary(
    text: str,
    summary_mode: Literal["short", "medium", "long", "auto"] = "medium",
    max_chunk_tokens: int = 200,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    summary_words = 150
) -> str:
    mgr = SummarizerManager()
    txt = normalize_text(text)
    chunks = chunk_text(txt, max_chunk_tokens)

    summaries = []
    for chunk in chunks:
        cnt = len(mgr.tokenizer.tokenize(chunk))
        mn, mx = (min_length, max_length) if min_length and max_length else scale_lengths(summary_mode, cnt)
        summaries.append(mgr.summarize_chunk(chunk, mn, mx))

    merged = " ".join(summaries)
    words = merged.split()
    if len(words) > summary_words:
        merged = " ".join(words[:150]) + "..."
    return merged
