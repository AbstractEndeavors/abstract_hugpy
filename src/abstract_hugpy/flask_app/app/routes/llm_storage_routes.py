from ..functions import *

llm_bp, logger = get_bp("llm_bp", __name__)

for name in ("httpx", "httpcore", "huggingface_hub", "filelock", "urllib3"):
    logging.getLogger(name).setLevel(logging.INFO)

# ──────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────
@llm_bp.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "storage_root": str(settings.storage_root),
        "manifest_path": str(settings.manifest_path),
    })


@llm_bp.route("/llm/peers", methods=["GET"])
def peers():
    return jsonify(list_peers())


@llm_bp.route("/models", methods=["GET"])
def list_models():
    manifest = get_models_dict(dict_return=True)
    output = []
    for key, model in manifest.items():
        model = update_model_status(model)
        output.append(model)

    return jsonify(output)


@llm_bp.route("/models/<model_key>", methods=["GET"])
def get_model(model_key):
    manifest = get_models_dict(dict_return=True)
    logger.info(manifest)
    if model_key not in manifest:
        abort(404, description="Unknown model key.")
    model = manifest[model_key]
    return jsonify({"key": model_key, **model, **model_status(model)})


@llm_bp.route("/models/<model_key>/download", methods=["POST"])
def start_download(model_key):
    model = get_model_config(model_key,dict_return=True)
    if not model:
        abort(404, description="Unknown model key.")
    logger.info(model)
    body = request.get_json(silent=True) or {}
    job = job_store.create(model_key)
    start_cancellable_download(job, model, total_bytes=body.get("total_bytes"))
    return jsonify(job.to_dict())


@llm_bp.route("/jobs", methods=["GET"])
def list_jobs():
    return jsonify([job.to_dict() for job in job_store.all()])


@llm_bp.route("/jobs/<job_id>", methods=["GET"])
def get_job(job_id):
    job = job_store.get(job_id)
    if not job:
        abort(404, description="Unknown job ID.")
    return jsonify(job.to_dict())


@llm_bp.route("/jobs/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id):
    return jsonify(cancel_download(job_id))


@llm_bp.route("/llm/repos/download", methods=["POST"])
def download_repo():
    """Acquire any Hugging Face repo by hub_id without a pre-registered manifest entry.

    If register=True, the model is added to the manifest so it appears in the
    registry browser on the next refresh.
    """
    body = HFRepoDownloadRequest(**(request.get_json(silent=True) or {}))
    model = {
        "name": body.name or body.hub_id.split("/")[-1],
        "hub_id": body.hub_id,
        "framework": body.framework,
        "task": body.task,
        "filename": body.filename,
        "include": body.include,
    }

    if body.register:
        model_key, _ = upsert_model(settings.manifest_path, model)
    else:
        from ..functions.llm_storage.utils.manifest import key_for_hub_id
        model_key = key_for_hub_id(body.hub_id)

    job = job_store.create(model_key)
    start_cancellable_download(job, model, total_bytes=body.total_bytes)
    return jsonify({**job.to_dict(), "model_key": model_key})


@llm_bp.route("/models/<model_key>", methods=["DELETE"])
def delete_model(model_key):
    manifest = get_models_dict(dict_return=True)
    if model_key not in manifest:
        abort(404, description="Unknown model key.")

    destination = route_destination(manifest.get(model_key))
    if not os.path.exists(destination):
        return jsonify({
            "deleted": False,
            "message": "Model is not installed.",
            "destination": str(destination),
        })

    shutil.rmtree(destination)
    return jsonify({"deleted": True, "destination": str(destination)})
