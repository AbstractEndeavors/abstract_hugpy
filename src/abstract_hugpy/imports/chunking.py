from .init_imports import List, Optional,re
from .module_imports import get_tiktoken
def get_encoder(model_name: str = "gpt-4", encoding_name: Optional[str] = None):
    """Return a tiktoken encoder for your model or encoding."""
    tiktoken = get_tiktoken()
    if encoding_name:
        return tiktoken.get_encoding(encoding_name)
    return tiktoken.encoding_for_model(model_name)

def count_tokens(text: str, encoder) -> int:
    """Count how many tokens `text` encodes to."""
    return len(encoder.encode(str(text)))

def recursive_chunk(
    text: str,
    desired_tokens: int,
    model_name: str = "gpt-4",
    separators: Optional[List[str]] = None,
    overlap: int = 0
) -> List[str]:
    """
    Split `text` into chunks as close to `desired_tokens` tokens as possible,
    preserving contiguous blocks via `separators`, and *only* splitting inside
    a block if it can’t possibly fit otherwise.
    
    Args:
        text: the full string to split
        desired_tokens: target token count per chunk (never exceed)
        encoder: a tiktoken encoder
        separators: list of splitters, from largest to smallest logical unit
        overlap: how many tokens to overlap between adjacent chunks
    """
    encoder = get_encoder(model_name)
    if separators is None:
        # from big (paragraphs) to small (words)
        separators = ["\n\n", "\n", r"(?<=[\.\?\!])\s", ", ", " "]

    # If it already fits, return it whole:
    if count_tokens(text, encoder) <= desired_tokens:
        return [text]

    # Try splitting by each separator in turn
    for sep in separators:
        # use regex split when the separator is a lookbehind pattern
        parts = re.split(sep, text) if sep.startswith("(?") else text.split(sep)
        if len(parts) > 1:
            chunks: List[str] = []
            current = ""
            current_tokens = 0

            for part in parts:
                part = part.strip()
                if not part:
                    continue
                part_tokens = count_tokens(part, encoder)

                # If this block alone is too big, recurse into it with the next-level separators
                if part_tokens > desired_tokens:
                    # flush current
                    if current:
                        chunks.extend(recursive_chunk(
                            current, desired_tokens, model_name, separators[1:], overlap
                        ))
                        current, current_tokens = "", 0
                    # now chunk the oversized block
                    chunks.extend(recursive_chunk(
                        part, desired_tokens, model_name, separators[1:], overlap
                    ))
                else:
                    # can we add it to the current chunk?
                    if current_tokens + part_tokens <= desired_tokens:
                        # include the separator back in
                        current = sep.join([current, part]) if current else part
                        current_tokens += part_tokens
                    else:
                        # flush current, start new
                        chunks.append(current)
                        current, current_tokens = part, part_tokens

            if current:
                chunks.append(current)
            return chunks

    # Fallback: pure token sliding window (the only time we’ll split “inside” a block)
    tokens = encoder.encode(text)
    stride = desired_tokens - overlap
    return [
        encoder.decode(tokens[i : i + desired_tokens])
        for i in range(0, len(tokens), stride)
    ]


