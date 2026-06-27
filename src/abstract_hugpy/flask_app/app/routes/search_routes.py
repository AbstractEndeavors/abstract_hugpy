### routes/search_routes.py

from ..functions import *
search_bp, logger = get_bp("search_bp", __name__)

# Single declared token source for ALL Hugging Face calls in this module.
# token=False FORCES anonymous (HF public search/metadata needs no auth); a
# stale ~/.cache/huggingface/token cannot poison the call. Set HF_TOKEN only
# for gated/private repos or higher rate limits. Everything below goes through
# this `api` object so there is exactly one place a token can come from.
api = HfApi(token=os.getenv("HF_TOKEN") or False)


# ── helpers ───────────────────────────────────────────────────────────────
def _free_bytes() -> int | None:
    """Headroom on the filesystem where downloads actually land (MODELS_DIR)."""
    try:
        probe = MODELS_DIR if os.path.exists(MODELS_DIR) else "/"
        return shutil.disk_usage(probe).free
    except OSError:
        return None


def _context_length(hub_id: str, files) -> int | None:
    if not any(f.path == "config.json" for f in files):
        return None
    try:
        cfg_path = api.hf_hub_download(hub_id, "config.json")   # tiny, cached
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        return cfg.get("max_position_embeddings") or cfg.get("n_positions")
    except Exception:
        return None


def _license_of(info) -> str | None:
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    try:
        return card.to_dict().get("license")
    except Exception:
        return getattr(card, "license", None)


# ── search: list available HF repos ────────────────────────────────────────
@search_bp.route("/search", methods=["GET"])
def search_models():
    q = request.args.get("q", "").strip()
    limit = request.args.get("limit", default=20, type=int)
    author = request.args.get("author")
    task = request.args.get("task")          # -> pipeline_tag
    library = request.args.get("library")    # -> filter
    sort = request.args.get("sort", default="last_modified")
    direction = request.args.get("direction", default=-1, type=int)  # client-side only now
    with_size = request.args.get("with_size", default="1") != "0"

    if sort not in ("last_modified", "downloads", "likes", "created_at"):
        sort = "last_modified"

    try:
        models = api.list_models(
            search=q or None,
            author=author,
            pipeline_tag=task or None,        # task is its own param
            filter=library or None,           # filter = library/tag
            sort=sort,
            limit=limit,
            full=False,
        )
    except Exception as exc:
        abort(502, description=f"Hugging Face request failed: {exc}")

    results = []
    for model in models:
        hub_id = model.modelId
        results.append(
            ModelSearchResult(
                hub_id=hub_id,
                author=getattr(model, "author", None),
                downloads=getattr(model, "downloads", None),
                likes=getattr(model, "likes", None),
                tags=getattr(model, "tags", []) or [],
                pipeline_tag=getattr(model, "pipeline_tag", None),
                library_name=getattr(model, "library_name", None),
                private=getattr(model, "private", None),
                total_bytes=model_size(hub_id) if with_size else None,   # #2
                last_modified=str(getattr(model, "last_modified", "")) or None,
            ).model_dump()
        )

    return jsonify(results)



# ── spec: per-repo detail + install options (lazy, on row expand) ───────────
@search_bp.route("/hf/spec", methods=["GET"])
def hf_spec():
    hub_id = request.args.get("hub_id")
    if not hub_id:
        abort(400, description="hub_id is required.")

    try:
        info = api.model_info(hub_id, files_metadata=True)
    except Exception as exc:
        abort(502, description=f"Hugging Face request failed: {exc}")

    files = [FileSpec(path=s.rfilename, size=s.size) for s in info.siblings]
    total = sum(f.size for f in files if f.size) or None
    free = _free_bytes()

    task = getattr(info, "pipeline_tag", None) or "text-generation"
    options = resolve_options(hub_id, task, files, free)

    num_params = getattr(getattr(info, "safetensors", None), "total", None)

    spec = ModelSpec(
        hub_id=hub_id,
        license=_license_of(info),
        gated=getattr(info, "gated", None),
        last_modified=str(getattr(info, "last_modified", "")) or None,
        total_bytes=total,
        num_params=num_params,
        context_length=_context_length(hub_id, files),
        gguf_quants=[o.id.split(":", 1)[1] for o in options.options
                     if o.id.startswith("gguf:")],
        files=files,
    )

    return jsonify({"spec": spec.model_dump(), "options": options.model_dump()})
