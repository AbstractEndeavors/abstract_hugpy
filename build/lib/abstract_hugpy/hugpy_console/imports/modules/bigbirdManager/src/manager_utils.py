from ..imports import *
from .manager import BigBirdManager
from ...flanManager import FlanManager
def get_content_length(text: str) -> List[int]:
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


def generate_with_bigbird(
    text: str,
    task: Optional[str] = None,
    manager: Optional[BigBirdManager] = None,
) -> str:
    """
    Use a persistent BigBirdManager to generate a summary or title.
    """
    manager = manager or BigBirdManager()
    torch = manager.torch
    tokenizer, model = manager.tokenizer, manager.model

    try:
        if task in {"title", "caption", "description"}:
            prompt = f"Generate a concise, SEO-optimized {task} for the following content:\n{text[:1000]}"
            max_len = 200
        else:
            prompt = f"Summarize the following content into a 100-150 word SEO-optimized abstract:\n{text[:4000]}"
            max_len = 300

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(manager.device)
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

def refine_with_gpt(
    full_text: str,
    task: Optional[str] = "abstract",
    generator_fn: Optional[Callable] = None,
    flan_manager: Optional[FlanManager] = None,
) -> str:
    """
    Two-stage refinement:
      1. BigBird drafts a summary.
      2. FlanManager or custom generator polishes the output.
    """
    prompt = generate_with_bigbird(full_text, task=task)
    if not prompt:
        return ""

    lengths = get_content_length(full_text)
    min_len, max_len = (lengths + [100, 200])[:2]

    try:
        if generator_fn is None:
            flan_manager = flan_manager or FlanManager()
            pipeline = flan_manager.pipeline
            if not pipeline:
                flan_manager._safe_preload()
                pipeline = flan_manager.pipeline
            generator_fn = pipeline

        out = generator_fn(prompt, min_length=min_len, max_length=max_len, num_return_sequences=1)
        if isinstance(out, list) and "generated_text" in out[0]:
            return out[0]["generated_text"].strip()
        return str(out)

    except Exception as e:
        logger.error(f"Refinement failed: {e}")
        return prompt
