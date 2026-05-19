from .imports import (
    get_torch,
    get_transformers,
    SingletonMeta,
    Callable,
    List,
    Optional,
    SingletonMeta,
    DEFAULT_PATHS
    )

DEFAULT_BIGBIRD_PATH: str = DEFAULT_PATHS["bigbird"]


# FIX: model was reloaded from disk on every call to generate_with_bigbird.
# LEDForConditionalGeneration is large; singleton it like every other model here.
class LEDModelManager(metaclass=SingletonMeta):
    
    def __init__(self, model_dir: str = DEFAULT_BIGBIRD_PATH):
        if not hasattr(self, "initialized"):
            self.initialized = True
            self.model_dir = model_dir
            self.tokenizer = get_transformers('LEDTokenizer').from_pretrained(model_dir)
            self.model = get_transformers('LEDForConditionalGeneration').from_pretrained(model_dir)

    def generate(self, prompt: str, max_length: int = 200) -> str:
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_beams=5,
                early_stopping=True,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def get_led_manager(model_dir: Optional[str] = None):
    return get_transformers('LEDModelManager')(model_dir or DEFAULT_BIGBIRD_PATH)


# ------------------------------------------------------------------------------
# 4. BIGBIRD-BASED "GPT"-STYLE REFINEMENT
# ------------------------------------------------------------------------------

def get_content_length(text: str) -> List[int]:
    """
    Given a text snippet containing hints like "into a X-Y word ...",
    extract numerical values and multiply by 10 to get a rough
    min/max length estimate for generation.

    E.g.: "Generate into a 5-10 word title" -> [50, 100]
    """
    for marker in ["into a "]:
        if marker in text:
            text = text.split(marker, 1)[1]
            break
    for ending in [" word", " words"]:
        if ending in text:
            text = text.split(ending, 1)[0]
            break

    numbers = []
    for part in text.split("-"):
        digits = "".join(ch for ch in part if ch.isdigit())
        numbers.append(int(digits) * 10 if digits else None)
    return [n for n in numbers if n is not None]


def generate_with_bigbird(
    text: str,
    task: str = "title",
    model_dir: Optional[str] = None,
) -> str:
    """
    Use LED (Longformer-Encoder-Decoder) to generate a prompt or partial summary.

    Args:
        text (str): Input text to condition on.
        task (str): One of {"title", "caption", "description", "abstract"}.
        model_dir (str): Override HuggingFace checkpoint path.

    Returns:
        str: The generated text from LED.
    """
    try:
        manager = get_led_manager(model_dir)

        if task in {"title", "caption", "description"}:
            prompt = (
                f"Generate a concise, SEO-optimized {task} "
                f"for the following content: {text[:1000]}"
            )
            max_length = 200
        else:
            prompt = (
                f"Summarize the following content into a 100-150 word "
                f"SEO-optimized abstract: {text[:4000]}"
            )
            max_length = 300

        return manager.generate(prompt, max_length=max_length)

    except Exception as e:
        print(f"Error in BigBird processing: {e}")
        return ""


def refine_with_gpt(
    full_text: str,
    task: str = "title",
    generator_fn: Optional[Callable] = None,
) -> str:
    """
    Two-step refinement:
      1) generate_with_bigbird() crafts a prompt/initial summary.
      2) generator_fn() (causal LM) refines it.

    Args:
        full_text (str): The text to refine.
        task (str): One of {"title", "caption", "description", "abstract"}.
        generator_fn (callable): Takes (prompt, min_length, max_length, num_return_sequences),
            returns list of dicts with "generated_text".

    Returns:
        str: The final refined text.
    """
    if generator_fn is None:
        raise ValueError(
            "generator_fn is required (e.g. pipeline('text-generation') or a custom callable)."
        )

    prompt = generate_with_bigbird(full_text, task=task)
    if not prompt:
        return ""

    lengths = get_content_length(full_text)
    min_length, max_length = 100, 200
    if lengths:
        min_length = lengths[0]
        max_length = lengths[-1] if len(lengths) > 1 else max_length

    out = generator_fn(prompt, min_length=min_length, max_length=max_length, num_return_sequences=1)
    if isinstance(out, list) and out and "generated_text" in out[0]:
        return out[0]["generated_text"].strip()
    return ""
