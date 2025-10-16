## BigBird + GPT-style Refiner (Manager-compatible version)
from .imports import *
logger = logging.getLogger("BigBirdRefiner")

# --------------------------------------------------------------------------
# Utility: extract length hints from text
# --------------------------------------------------------------------------
def get_content_length(text: str) -> List[int]:
    """
    Parse hints like "into a X-Y word ..." and return rough [min, max] values ×10.
    """
    for marker in ["into a "]:
        if marker in text:
            text = text.split(marker, 1)[1]
            break
    for ending in [" word", " words"]:
        if ending in text:
            text = text.split(ending, 1)[0]
            break

    nums = []
    for part in text.split("-"):
        digits = "".join(ch for ch in part if ch.isdigit())
        nums.append(int(digits) * 10 if digits else None)
    return [n for n in nums if n is not None]


# --------------------------------------------------------------------------
# Step 1: LED (BigBird) summarization
# --------------------------------------------------------------------------
def generate_with_bigbird(
    text: str,
    task: str = None,
    model_dir: str = None,
) -> str:
    """
    Generate a summary or title using LED (Longformer-Encoder-Decoder).
    Uses lazy imports so transformers aren’t loaded until needed.
    """
    model_dir = model_dir or DEFAULT_PATHS.get("bigbird")
    torch = get_torch()
    LEDTokenizer = get_LEDTokenizer()
    LEDForConditionalGeneration = get_LEDForConditionalGeneration()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        tokenizer = LEDTokenizer.from_pretrained(model_dir)
        model = LEDForConditionalGeneration.from_pretrained(model_dir).to(device)

        if task in {"title", "caption", "description"}:
            prompt = f"Generate a concise, SEO-optimized {task} for the following content:\n{text[:1000]}"
            max_len = 200
        else:
            prompt = f"Summarize the following content into a 100-150 word SEO-optimized abstract:\n{text[:4000]}"
            max_len = 300

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_len,
                num_beams=5,
                early_stopping=True,
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    except Exception as e:
        logger.error(f"BigBird generation failed: {e}")
        return ""


# --------------------------------------------------------------------------
# Step 2: Refinement with FLAN-T5 or custom generator
# --------------------------------------------------------------------------
def refine_with_gpt(
    full_text: str,
    task: str = None,
    generator_fn: Optional[Callable] = None,
    model_dir: str = None,
) -> str:
    """
    Two-stage refinement:
      1. BigBird drafts a summary.
      2. A seq2seq generator (e.g. FLAN-T5) polishes the language.
    """
    task = task or "abstract"
    model_dir = model_dir or DEFAULT_PATHS.get("flan")
    prompt = generate_with_bigbird(full_text, task=task)
    if not prompt:
        return ""

    lengths = get_content_length(full_text)
    min_len, max_len = (lengths + [100, 200])[:2]

    try:
        if generator_fn is None:
            AutoTokenizer = get_AutoTokenizer()
            AutoModelForSeq2SeqLM = get_AutoModelForSeq2SeqLM()
            tok = AutoTokenizer.from_pretrained(model_dir)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
            generator_fn = get_pipeline()(
                "text2text-generation", model=mdl, tokenizer=tok
            )

        out = generator_fn(prompt, min_length=min_len, max_length=max_len, num_return_sequences=1)
        if isinstance(out, list) and "generated_text" in out[0]:
            return out[0]["generated_text"].strip()
        return str(out)

    except Exception as e:
        logger.error(f"Refinement failed: {e}")
        return prompt
