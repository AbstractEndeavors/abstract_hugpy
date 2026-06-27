from .imports import *

# ---------------------------------------------------------------------------
# Request builders — one per (framework, task).
# ---------------------------------------------------------------------------

def _build_chat_request(kwargs: Dict[str, Any], model_key: str) -> ChatRequest:
    out: Dict[str, Any] = {"model_key": model_key}
    file = kwargs.get("file")

    if "messages" in kwargs:
        if file:
            raise ValueError(
                "file alongside 'messages' isn't wired on the chat path; "
                "send 'prompt' + 'file'"
            )
        out["messages"] = kwargs["messages"]
    elif "prompt" in kwargs:
        prompt = kwargs["prompt"] or ""
        if file:
            media = derive_media_type(file)
            if media not in ("text", "document", "code"):
                raise ValueError(
                    f"text-generation can't consume a {media!r} file "
                    f"({os.path.basename(file)}); route it to the matching model"
                )
            content = read_from_file(file)
            prompt = f"{prompt}\n------ {os.path.basename(file)} ------\n{content}"
        out["messages"] = [{"role": "user", "content": prompt}]
    else:
        raise ValueError(
            "chat request needs either 'messages' or 'prompt'; "
            f"got keys: {sorted(kwargs)}"
        )

    for k in ("max_new_tokens", "temperature", "top_p", "do_sample", "request_id", "unbounded"):
        if k in kwargs:
            out[k] = kwargs[k]
    out.setdefault("request_id", make_request_id())
    # Default chat to unbounded so the runner keeps generating until the model
    # naturally stops, instead of truncating at a single token cap. Callers can
    # still force a bounded response with unbounded=False / a max_new_tokens cap.
    if "unbounded" not in out and not kwargs.get("max_new_tokens"):
        out["unbounded"] = True
    return ChatRequest(**out)


def _build_vision_request(kwargs: Dict[str, Any], model_key: str) -> VisionRequest:
    image_path = kwargs.get("image_path") or kwargs.get("file")
    if image_path is None:
        raise ValueError(
            "vision request needs 'image_path' or 'file'; "
            f"got keys: {sorted(kwargs)}"
        )

    out: Dict[str, Any] = {
        "model_key": model_key,
        "image_path": image_path,
        "request_id": kwargs.get("request_id", make_request_id()),
    }
    for k in ("prompt", "max_new_tokens", "max_tokens"):
        if k in kwargs:
            out[k] = kwargs[k]
    return VisionRequest(**out)


def _build_whisper_request(kwargs: Dict[str, Any], model_key: str) -> TranscribeRequest:
    file_path = kwargs.get("audio_path") or kwargs.get("file")
    if file_path is None:
        raise ValueError(
            "whisper request needs 'audio_path' or 'file'; "
            f"got keys: {sorted(kwargs)}"
        )
    return TranscribeRequest(
        model_key=model_key,
        file_path=file_path,
        capture_frames=kwargs.get("capture_frames", False),
        request_id=kwargs.get("request_id", make_request_id()),
    )


def _build_summarize_request(kwargs: Dict[str, Any], model_key: str) -> "SummarizeRequest":
    text = kwargs.get("text") or kwargs.get("prompt")
    if text is None and kwargs.get("file"):
        text = read_from_file(kwargs["file"])
    if text is None:
        raise ValueError(
            "summarize request needs 'text', 'prompt', or 'file'; "
            f"got keys: {sorted(kwargs)}"
        )

    out: Dict[str, Any] = {
        "model_key": model_key,
        "text": text,
        "request_id": kwargs.get("request_id", make_request_id()),
    }
    for k in (
        "preset", "summary_mode", "input_policy",
        "max_chunk_tokens", "min_length", "max_length",
        "do_sample", "min_input_words",
        "consolidation_min_length", "consolidation_max_length",
        "max_output_words",
    ):
        if k in kwargs:
            out[k] = kwargs[k]
    return SummarizeRequest(**out)


def _texts_from_kwargs(kwargs: Dict[str, Any]) -> list[str]:
    """Shared text extraction: texts | text | prompt | file -> list[str].

    Used by both embed builders. Returns a list even for single-string
    input so the runner doesn't have to branch.
    """
    raw = kwargs.get("texts") or kwargs.get("text") or kwargs.get("prompt")
    if raw is None and kwargs.get("file"):
        raw = read_from_file(kwargs["file"])
    if raw is None:
        raise ValueError(
            "embed request needs 'texts', 'text', 'prompt', or 'file'; "
            f"got keys: {sorted(kwargs)}"
        )
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list) and all(isinstance(t, str) for t in raw):
        return list(raw)
    raise TypeError(
        f"embed input must be str or list[str], got {type(raw).__name__}"
    )


def _build_embed_request(kwargs: Dict[str, Any], model_key: str) -> EmbedRequest:
    return EmbedRequest(
        model_key=model_key,
        request_id=kwargs.get("request_id", make_request_id()),
        texts=_texts_from_kwargs(kwargs),
        normalize=kwargs.get("normalize", True),
        batch_size=kwargs.get("batch_size", 32),
    )


def _build_similarity_request(kwargs: Dict[str, Any], model_key: str) -> EmbedRequest:
    """sentence-similarity needs a second set of texts to compare against."""
    other_raw = (
        kwargs.get("other_texts")
        or kwargs.get("other_text")
        or kwargs.get("compare_to")
    )
    if other_raw is None:
        raise ValueError(
            "sentence-similarity needs 'other_texts', 'other_text', or 'compare_to' "
            f"in addition to 'texts'/'text'/'prompt'/'file'; got keys: {sorted(kwargs)}"
        )
    if isinstance(other_raw, str):
        other_texts = [other_raw]
    elif isinstance(other_raw, list) and all(isinstance(t, str) for t in other_raw):
        other_texts = list(other_raw)
    else:
        raise TypeError(
            f"other_texts must be str or list[str], got {type(other_raw).__name__}"
        )

    return EmbedRequest(
        model_key=model_key,
        request_id=kwargs.get("request_id", make_request_id()),
        texts=_texts_from_kwargs(kwargs),
        other_texts=other_texts,
        normalize=kwargs.get("normalize", True),
        batch_size=kwargs.get("batch_size", 32),
    )


# ---------------------------------------------------------------------------
# Registries — single source of truth.
# ---------------------------------------------------------------------------

MODEL_REQUEST_BUILDERS: Dict[Tuple[str, str], Callable[[Dict[str, Any], str], BaseModel]] = {
    ("transformers", "text-generation"):              _build_chat_request,
    ("llama_cpp",    "text-generation"):              _build_chat_request,
    ("transformers", "image-text-to-text"):           _build_vision_request,
    ("transformers", "automatic-speech-recognition"): _build_whisper_request,
    ("transformers", "text-summarization"):                _build_summarize_request,
    ("transformers", "text2text-generation"):         _build_summarize_request,
    ("transformers", "feature-extraction"):           _build_embed_request,
    ("transformers", "sentence-similarity"):          _build_similarity_request,
}





