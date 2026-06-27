"""Model resolution — single source of truth.

Everything in dispatch reads from `Resolution`, which is built exactly
once per request by `resolve()`. No downstream layer is allowed to
re-derive task, framework, builder, or runner_cls from kwargs — if it
needs any of those, it reads them off the Resolution object.

Adding a new (framework, task) pair:
    1. Implement a runner class conforming to the Runner protocol.
    2. Add a row to FRAMEWORK_RUNNERS.
    3. Add a row to MODEL_REQUEST_BUILDERS.
    4. (Optional) Add a row to TASK_DEFAULTS if there's a sensible
       default model for "task only" callers.

Adding a new model:
    Add a row to MODEL_REGISTRY (in models_dict.py). validate_registry()
    will fail at import time if (framework, primary_task) or any
    (framework, task) in cfg.tasks isn't registered.
"""

from .imports import *
from .categories import *
from .assure_model_key import assure_model_key
# ---------------------------------------------------------------------------
# Peer placement (System A) — placement.json delegation.
#
# resolve() calls peer_for() on EVERY request, so these must always be defined,
# even when no placement file exists. A missing/empty placement.json means
# "everything runs locally" — peer_for() returns None and resolve() falls
# through to the local runner. (This block was historically edited only on the
# deployed server and never committed, so the repo's resolve() raised NameError
# on peer_for for every request — which looked like 'no compute allocated'.)
# ---------------------------------------------------------------------------
try:
    PLACEMENT_PATH
except NameError:
    PLACEMENT_PATH = os.path.join(PROJECTS_HOME, "placement.json")


class Peer(BaseModel):
    name: str
    base_url: str              # http://192.168.1.x:PORT — the peer's flask app
    role: str = "compute"
    status: str = "unknown"    # filled by a health ping


# Placement registry: "model_key::task" -> worker name | "local" | absent.
# Empty default = everything runs locally. _load_placement() populates these.
_placement: Dict[str, str] = {}
_peers: Dict[str, Peer] = {}


def _load_placement(path: Optional[str] = None) -> None:
    """Populate _placement/_peers from placement.json. Explicit call, not
    import-time magic — so a missing/empty file means 'all local', never a
    crash."""
    global _placement, _peers
    data = safe_load_from_json(path or PLACEMENT_PATH) or {}
    _placement = data.get("placement", {}) or {}
    _peers = {name: Peer(**cfg) for name, cfg in (data.get("peers", {}) or {}).items()}


def peer_for(model_key: str, task: str) -> Optional[Peer]:
    """Return the Peer that should serve (model_key, task), or None for local.

    Looks up "model_key::task" in the placement map; "local"/absent -> None.
    """
    name = _placement.get(f"{model_key}::{task}")
    if name in (None, "local"):
        return None
    return _peers.get(name)


# Load once at import; safe no-op when placement.json is absent.
try:
    _load_placement()
except Exception as exc:  # never let placement config break resolution
    logger.warning("placement.json load failed (%s); all models run local", exc)


# ---------------------------------------------------------------------------
# resolve_model_key — picks the model. Default-resolution chain only.
# Does NOT pick task; that's resolve()'s job.
# ---------------------------------------------------------------------------
def make_remote_runner(peer, framework, task):
    local_cls = FRAMEWORK_RUNNERS[(framework, task)]   # borrow result_type

    class RemoteRunner:
        request_type = local_cls.request_type
        result_type  = local_cls.result_type
        def __init__(self, cfg):
            self.cfg = cfg
            self.model_key = cfg.model_key
        async def run(self, req):
            import asyncio
            from abstract_apis import postRequest
            payload = {"delegated": True, "task": task, **req.model_dump()}
            # postRequest is sync; keep the event loop free
            data = await asyncio.to_thread(
                postRequest,
                url=peer.base_url,
                endpoint="api/llm/execute",
                data=payload,
                timeout=self.cfg.timeout_s or 3600,
            )
            return self.result_type.model_validate(data)
    return RemoteRunner
def resolve_model_key(
    *,
    model_key: Optional[str] = None,
    file: Optional[str] = None,
    media_type: Optional[str] = None,
    task: Optional[str] = None,
) -> str:
    """Pick a model_key via explicit resolution chain.

    Order: explicit model_key > explicit task > explicit media_type
           > file -> media_type > chat default.

    `task`, when given alongside `model_key`, is validated against
    cfg.tasks. When given alone, it picks TASK_DEFAULTS[task].
    """
    if task is not None and task not in KNOWN_TASKS_REGISTRY:
        print(f"task:{task} in KNOWN_TASKS_REGISTRY:{KNOWN_TASKS_REGISTRY}")
        raise KeyError(
            f"Unknown task={task!r}; known: {sorted(KNOWN_TASKS_REGISTRY)}"
        )
    
    if model_key is not None:
        model_key = assure_model_key(model_key)
        if not model_key:
            raise KeyError(
                f"Unknown model_key={model_key!r}; "
                f"known: {sorted(MODEL_REGISTRY.keys())}"
            )
        if task is not None and task not in MODEL_REGISTRY[model_key].tasks:
            raise ValueError(
                f"Model {model_key!r} does not support task={task!r}; "
                f"supported: {sorted(MODEL_REGISTRY[model_key].tasks)}"
            )
        logger.debug("resolve_model_key: explicit key=%s task=%s", model_key, task)
        return model_key

    if task is not None:
        print(f"task:{task} in TASK_DEFAULTS:{TASK_DEFAULTS}")
        inferred = TASK_DEFAULTS.get(task)
        if inferred is None:
            raise KeyError(
                f"No default model for task={task!r}; "
                f"tasks with defaults: {sorted(TASK_DEFAULTS)}"
            )
        if inferred not in MODEL_REGISTRY:
            raise KeyError(
                f"Task default {inferred!r} for {task!r} not in MODEL_REGISTRY:{MODEL_REGISTRY}"
            )
        if task not in MODEL_REGISTRY[inferred].tasks:
            raise ValueError(
                f"Task default {inferred!r} for {task!r} does not list "
                f"{task!r} in cfg.tasks={sorted(MODEL_REGISTRY[inferred].tasks)!r}"
            )
        logger.debug("resolve_model_key: task=%s -> key=%s", task, inferred)
        return inferred

    if media_type is None and file is not None:
        if not os.path.exists(file):
            raise FileNotFoundError(
                f"resolve_model_key: file does not exist: {file!r}"
            )
        media_type = derive_media_type(file)
        logger.debug("resolve_model_key: file=%s -> media=%s", file, media_type)

    if media_type is not None:
        inferred = MEDIA_DEFAULTS.get(media_type)
        if inferred is None:
            raise KeyError(
                f"No default model for media_type={media_type!r}; "
                f"known: {sorted(MEDIA_DEFAULTS)}"
            )
        if inferred not in MODEL_REGISTRY:
            raise KeyError(
                f"Media default {inferred!r} for {media_type!r} "
                f"not in MODEL_REGISTRY"
            )
        logger.debug("resolve_model_key: media=%s -> key=%s", media_type, inferred)
        return inferred

    if DEFAULT_CHAT_MODEL not in MODEL_REGISTRY:
        raise KeyError(
            f"DEFAULT_CHAT_MODEL={DEFAULT_CHAT_MODEL!r} not in MODEL_REGISTRY"
        )
    logger.debug("resolve_model_key: fallback to chat default=%s", DEFAULT_CHAT_MODEL)
    return DEFAULT_CHAT_MODEL


# ---------------------------------------------------------------------------
# resolve — the only function that maps kwargs -> Resolution.
# ---------------------------------------------------------------------------

def resolve(prompt_kwargs: Dict[str, Any]) -> Resolution:
    """Build a Resolution from request kwargs. One call site for all routing.

    `task`, if given by the caller, wins over cfg.primary_task. This is the
    single rule that the old dispatch broke in three different places.
    """
    requested_task = prompt_kwargs.get("task")

    model_key = resolve_model_key(
        model_key=prompt_kwargs.get("model_key"),
        file=prompt_kwargs.get("file"),
        media_type=prompt_kwargs.get("media_type"),
        task=requested_task,
    )

    cfg = MODEL_REGISTRY[model_key]

    task = requested_task or cfg.primary_task
    if isinstance(task, (list, tuple)):       # primary_task must be scalar
        task = task[0]

    framework = cfg.framework
    if isinstance(framework, (list, tuple)):  # framework must be scalar too
        framework = framework[0]

    # TEMP diagnostic — shows which field is malformed in the registry row.
    logger.info("resolve types: model=%s framework=%r task=%r primary=%r tasks=%r",
                model_key, cfg.framework, task, cfg.primary_task, cfg.tasks)

    if task not in cfg.tasks:
        raise ValueError(
            f"Model {model_key!r} does not support task={task!r}; "
            f"supported: {sorted(cfg.tasks)}"
        )

    key = (framework, task)

    builder = MODEL_REQUEST_BUILDERS.get(key)
    if builder is None:
        raise KeyError(
            f"No request builder for {key!r}; model={model_key!r}, "
            f"known: {sorted(MODEL_REQUEST_BUILDERS)}"
        )
    force_local = prompt_kwargs.pop("_force_local", False)
    peer = None if force_local else peer_for(model_key, task)

    if peer:
        # placement.json delegation: run this (model, task) on a remote peer.
        runner_cls = make_remote_runner(peer, framework, task)
    else:
        runner_cls = FRAMEWORK_RUNNERS.get(key)
        if runner_cls is None:
            raise KeyError(
                f"No runner for {key!r}; model={model_key!r}, "
                f"known: {sorted(FRAMEWORK_RUNNERS)}"
            )

    logger.debug(
        "resolve: model=%s framework=%s task=%s (requested=%s primary=%s)",
        model_key, framework, task, requested_task, cfg.primary_task,
    )

    return Resolution(
        model_key=model_key,
        framework=framework,          # scalar, not cfg.framework
        task=task,
        cfg=cfg,
        builder=builder,
        runner_cls=runner_cls,
        cache_key=(model_key, task),
    )

# ---------------------------------------------------------------------------
# validate_registry — fail at import time, not on first request.
# ---------------------------------------------------------------------------

def validate_registry() -> None:
    """Walk MODEL_REGISTRY and assert every entry can actually be served.

    Two checks per model:
      1. (framework, primary_task) has a runner registered.
      2. Every task in cfg.tasks has a runner AND a builder registered.

    Raises RuntimeError listing ALL broken entries — not just the first —
    so a single import gives you the full list of registry bugs to fix.
    """
    errors: list[str] = []

    for model_key, cfg in MODEL_REGISTRY.items():
        primary_key = (cfg.framework, cfg.primary_task)
        if primary_key not in FRAMEWORK_RUNNERS:
            errors.append(
                f"  {model_key}: primary_task={cfg.primary_task!r} on "
                f"framework={cfg.framework!r} has no runner registered"
            )

        for task in cfg.tasks:
            task_key = (cfg.framework, task)
            if task_key not in FRAMEWORK_RUNNERS:
                errors.append(
                    f"  {model_key}: task={task!r} in cfg.tasks on "
                    f"framework={cfg.framework!r} has no runner registered"
                )
            if task_key not in MODEL_REQUEST_BUILDERS:
                errors.append(
                    f"  {model_key}: task={task!r} in cfg.tasks on "
                    f"framework={cfg.framework!r} has no request builder registered"
                )

    if errors:
        raise RuntimeError(
            f"MODEL_REGISTRY validation failed ({len(errors)} issues):\n"
            + "\n".join(errors)
            + f"\n\nRegistered runners:  {sorted(FRAMEWORK_RUNNERS)}"
            + f"\nRegistered builders: {sorted(MODEL_REQUEST_BUILDERS)}"
        )

    logger.info(
        "validate_registry: ok — %d models, %d runner pairs, %d builder pairs",
        len(MODEL_REGISTRY), len(FRAMEWORK_RUNNERS), len(MODEL_REQUEST_BUILDERS),
    )


# Run at import time. If the registry is bad, fail loudly here — not
# halfway through a user's request.
validate_registry()
