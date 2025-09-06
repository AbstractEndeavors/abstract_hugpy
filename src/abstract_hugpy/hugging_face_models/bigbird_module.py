import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LEDTokenizer,
    LEDForConditionalGeneration
)
# ------------------------------------------------------------------------------
# 4. BIGBIRD-BASED “GPT”-STYLE REFINEMENT
# ------------------------------------------------------------------------------
def get_content_length(text: str) -> list[int]:
    """
    Given a text snippet containing hints like "into a X-Y word ...",
    extract numerical values (X, Y) and multiply by 10 to get a rough
    min/max length estimate for generation.

    E.g.: "Generate into a 5-10 word title" → returns [50, 100].

    Args:
        text (str): Instructional snippet containing numeric hints.

    Returns:
        list[int]: [min_length*10, max_length*10] (if found), else empty list.
    """
    # Look for patterns like "into a {num}-{num} word"
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
    # Filter out None
    return [n for n in numbers if n is not None]


def generate_with_bigbird(
    text: str,
    task: str = "title",
    model_dir: str = "allenai/led-base-16384"
) -> str:
    """
    Use LED (Longformer-Encoder-Decoder) to generate a prompt or partial summary.
    Typically called internally by refine_with_gpt().

    Args:
        text (str): Input text to condition on.
        task (str): One of {"title", "caption", "description", "abstract"}. Defaults to "title".
        model_dir (str): HuggingFace checkpoint for LED. Defaults to "allenai/led-base-16384".

    Returns:
        str: The generated text from LED.
    """
    try:
        tokenizer = LEDTokenizer.from_pretrained(model_dir)
        model = LEDForConditionalGeneration.from_pretrained(model_dir)

        # Build a task-specific prompt
        if task in {"title", "caption", "description"}:
            prompt = f"Generate a concise, SEO-optimized {task} for the following content: {text[:1000]}"
        else:
            # Defaults to an "abstract"/summary style prompt
            prompt = f"Summarize the following content into a 100-150 word SEO-optimized abstract: {text[:4000]}"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=200 if task in {"title", "caption"} else 300,
                num_beams=5,
                early_stopping=True
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error in BigBird processing: {e}")
        return ""


def refine_with_gpt(
    full_text: str,
    task: str = None,
    generator_fn=None
) -> str:
    """
    A two‐step “refinement” that:
      1) Calls generate_with_bigbird(...) to craft a prompt or initial summary.
      2) Passes that prompt into a causal‐LM generator (e.g. GPT2, GPT-Neo, or a custom 'generator' function).

    Args:
        full_text (str): The text to refine.
        task (str): One of {"title", "caption", "description", "abstract"}. Defaults to "title".
        generator_fn (callable): A text‐generation function that takes (prompt, min_length, max_length), returns a list of dicts with "generated_text".

    Returns:
        str: The final refined text.
    """
    if not generator_fn:
        raise ValueError("You must supply a generator_fn (e.g. pipeline('text-generation') or a custom function).")

    # Step 1: Let BigBird draft a prompt or partial summary
    prompt = generate_with_bigbird(full_text, task=task)
    if not prompt:
        return ""

    # Step 2: Parse length hints from full_text and fallback to defaults
    lengths = get_content_length(full_text)
    min_length, max_length = 100, 200
    if lengths:
        # lengths may be [min_hint, max_hint], if both present
        min_length = lengths[0] if len(lengths) > 0 else min_length
        max_length = lengths[-1] if len(lengths) > 1 else max_length

    # Step 3: Run the generator function on the prompt
    # Expect generator_fn to return a list of dicts like: [{"generated_text": "..."}, ...]
    out = generator_fn(prompt, min_length=min_length, max_length=max_length, num_return_sequences=1)
    if isinstance(out, list) and "generated_text" in out[0]:
        return out[0]["generated_text"].strip()
    else:
        return ""
