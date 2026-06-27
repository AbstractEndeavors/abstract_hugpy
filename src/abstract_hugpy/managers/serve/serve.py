# imports/apis/serve.py  (or managers/serve/serve.py — see note at bottom)
"""
Serving abstraction. One model, one resolved ServeSpec; how it actually gets
served is a pluggable driver keyed by an explicit mode — not inferred from
whether cfg.port happens to be set.

    mode = off       -> not served externally; in-process runner handles it
    mode = systemd   -> a dedicated llama-server under its own unit (always-on)
    mode = swap      -> an entry in one shared llama-swap proxy (on-demand)

Why this shape:
    - registry, not port-sniffing: serve_spec_for() reads MODEL_REGISTRY + env,
      the same way resolve_model_source() does. cfg.port stops meaning four
      things at once.
    - driver registry, like vision_backends.register_backend and
      FRAMEWORK_RUNNERS: adding a serving mechanism is one @register_serve_driver.
    - plans, not callbacks: a driver returns a ServePlan (writes + commands).
      Dry-run renders it for the UI; apply() drains it. The run()/write()
      callables are the seam for local-vs-remote (ssh to a peer) execution.

Caters to either deployment with one env var:
    DEFAULT_SERVE_MODE=systemd   # every llama_cpp model gets its own unit
    DEFAULT_SERVE_MODE=swap      # everything routes through one llama-swap proxy
Per-model override lives in the overlay: cfg.extra["serve_mode"] = "swap".

os.path throughout, no pathlib.
"""

import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from .imports import *

logger = get_logFile(__name__)


# --------------------------------------------------------------------------- #
# explicit environment wiring                                                 #
# --------------------------------------------------------------------------- #

LLAMA_CPP_DIR = get_env_value("LLAMA_CPP_DIR") or "/srv/abstractendeavors/models/llama.cpp"
LLAMA_SERVER_BIN = get_env_value("LLAMA_SERVER_BIN") or os.path.join(
    LLAMA_CPP_DIR, "build", "bin", "llama-server"
)
LLAMA_SERVICE_USER = get_env_value("LLAMA_SERVICE_USER") or "solcatcher"
LLAMA_SERVICE_GROUP = get_env_value("LLAMA_SERVICE_GROUP") or "web"
SYSTEMD_UNIT_DIR = get_env_value("SYSTEMD_UNIT_DIR") or "/etc/systemd/system"
LLAMA_UNIT_PREFIX = get_env_value("LLAMA_UNIT_PREFIX") or "llama"

# llama-swap: one proxy, one config, one unit to bounce on config change.
LLAMA_SWAP_HOST = get_env_value("LLAMA_SWAP_HOST") or "127.0.0.1"
LLAMA_SWAP_PORT = int(get_env_value("LLAMA_SWAP_PORT") or 9292)
LLAMA_SWAP_CONFIG = get_env_value("LLAMA_SWAP_CONFIG") or "/etc/llama-swap/config.yaml"
LLAMA_SWAP_UNIT = get_env_value("LLAMA_SWAP_UNIT") or "llama-swap.service"
LLAMA_SWAP_TTL = int(get_env_value("LLAMA_SWAP_TTL") or 600)   # on-demand unload, seconds

DEFAULT_LLAMA_CTX = int(get_env_value("DEFAULT_LLAMA_CTX") or 4096)
DEFAULT_LLAMA_THREADS = int(get_env_value("DEFAULT_LLAMA_THREADS") or 6)
# GPU offload for llama-server. -1 = put every layer on the GPU (the right
# default for a CUDA-built llama-server); 0 = CPU only; N = first N layers.
# Without this flag llama-server defaults to CPU — the usual "GPU sits idle".
DEFAULT_LLAMA_NGL = int(get_env_value("DEFAULT_LLAMA_NGL") or -1)
DEFAULT_SERVE_MODE = get_env_value("DEFAULT_SERVE_MODE") or "systemd"

# Deterministic auto-port range for systemd units when a model has no explicit
# port. Wide span keeps hash collisions rare; an explicit cfg.port always wins.
LLAMA_PORT_BASE = int(get_env_value("LLAMA_PORT_BASE") or 7001)
LLAMA_PORT_SPAN = int(get_env_value("LLAMA_PORT_SPAN") or 4000)


class ServeMode(str, Enum):
    OFF = "off"
    SYSTEMD = "systemd"
    SWAP = "swap"


# --------------------------------------------------------------------------- #
# small resolvers                                                             #
# --------------------------------------------------------------------------- #

def _unit_slug(value):
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-._")
    return value or "model"


def _bare_host(value):
    value = value or LLAMA_HOST
    if "://" in value:
        value = value.split("://", 1)[1]
    return value.split("/", 1)[0] or "127.0.0.1"


def _effective_extra(model_key, cfg) -> dict:
    """cfg.extra with the persisted per-model UI override merged on top."""
    extra = dict(getattr(cfg, "extra", {}) or {})
    try:
        from .overrides import get_override
        extra.update(get_override(model_key))
    except Exception:  # overrides are optional; never break spec resolution
        pass
    return extra


def _ctx_for(cfg, model_key, extra=None):
    extra = extra if extra is not None else _effective_extra(model_key, cfg)
    if extra.get("llama_ctx"):
        return int(extra["llama_ctx"])
    mml = getattr(cfg, "model_max_length", None) or DEFAULT_LLAMA_CTX
    capped = min(int(mml), DEFAULT_LLAMA_CTX)
    if capped < int(mml):
        logger.info("%s: capping -c %s -> %s (set extra['llama_ctx'] to override)",
                    model_key, int(mml), capped)
    return capped


def _model_file_for(model_key, cfg):
    """Best-effort absolute GGUF path; '' if it can't be located yet. Never
    raises — a unit/config can be staged before the download lands."""
    try:
        source = resolve_model_source(model_key)
        if source and os.path.isfile(source):
            return source
    except (FileNotFoundError, KeyError) as exc:
        logger.info("%s: resolve_model_source unavailable (%s)", model_key, exc)
    if getattr(cfg, "filename", None):
        return os.path.join(get_model_path(model_key), cfg.filename)
    return ""


def _auto_port(model_key: str) -> int:
    """Deterministic per-model port so a GGUF model can get a systemd unit
    without hand-assigning one. Stable across runs and identical whether
    resolved for a single model (the HTTP runner) or the whole batch (install),
    so the endpoint and the unit always agree. Explicit cfg.port still wins.
    """
    import hashlib
    h = int(hashlib.sha1(model_key.encode("utf-8")).hexdigest(), 16)
    return LLAMA_PORT_BASE + (h % LLAMA_PORT_SPAN)


def _resolve_port(model_key, cfg) -> int:
    p = getattr(cfg, "port", None)
    try:
        if p is not None and int(p) > 0:
            return int(p)
    except (TypeError, ValueError):
        pass
    return _auto_port(model_key)


def _resolve_mode(cfg, extra=None) -> ServeMode:
    if getattr(cfg, "framework", None) != "llama_cpp":
        return ServeMode.OFF
    extra = extra if extra is not None else (getattr(cfg, "extra", {}) or {})
    explicit = extra.get("serve_mode")
    # systemd no longer falls back to off for a missing port — _resolve_port
    # auto-assigns a deterministic one.
    return ServeMode(explicit) if explicit else ServeMode(DEFAULT_SERVE_MODE)


# --------------------------------------------------------------------------- #
# schema                                                                      #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ServeSpec:
    model_key: str
    mode: ServeMode
    model_file: str = ""
    host: str = LLAMA_SWAP_HOST
    port: Optional[int] = None
    ctx_size: int = DEFAULT_LLAMA_CTX
    threads: int = DEFAULT_LLAMA_THREADS
    n_gpu_layers: int = DEFAULT_LLAMA_NGL
    always_on: bool = True
    ttl_seconds: Optional[int] = None
    user: str = LLAMA_SERVICE_USER
    group: str = LLAMA_SERVICE_GROUP
    working_directory: str = LLAMA_CPP_DIR
    server_bin: str = LLAMA_SERVER_BIN
    extra_args: Tuple[str, ...] = ()

    def __post_init__(self):
        if self.mode is ServeMode.OFF:
            return
        if not os.path.isabs(self.server_bin):
            raise ValueError(f"{self.model_key}: server_bin must be absolute")
        if self.mode is ServeMode.SYSTEMD:
            if not self.port or not 0 < self.port < 65536:
                raise ValueError(f"{self.model_key}: systemd needs a valid port, got {self.port!r}")
            if not self.model_file:
                logger.info("%s: systemd unit will reference an unresolved model path", self.model_key)
        if self.ctx_size < 1 or self.threads < 1:
            raise ValueError(f"{self.model_key}: ctx_size and threads must be >= 1")

    @property
    def unit_name(self) -> str:
        return _unit_slug(f"{LLAMA_UNIT_PREFIX}-{self.model_key}")

    @property
    def swap_name(self) -> str:
        return _unit_slug(self.model_key)


def serve_spec_for(model_key=None, *, cfg=None) -> ServeSpec:
    cfg = cfg if cfg is not None else get_model_config(model_key)
    model_key = model_key or cfg.model_key or cfg.name
    extra = _effective_extra(model_key, cfg)   # cfg.extra + persisted UI override
    mode = _resolve_mode(cfg, extra)

    if mode is ServeMode.SWAP:
        host, port = LLAMA_SWAP_HOST, LLAMA_SWAP_PORT
    elif mode is ServeMode.SYSTEMD:
        host, port = _bare_host(getattr(cfg, "host", None)), _resolve_port(model_key, cfg)
    else:
        host = _bare_host(getattr(cfg, "host", None))
        port = int(cfg.port) if getattr(cfg, "port", None) else None

    return ServeSpec(
        model_key=model_key,
        mode=mode,
        model_file=_model_file_for(model_key, cfg),
        host=host,
        port=port,
        ctx_size=_ctx_for(cfg, model_key, extra),
        threads=int(extra.get("threads") or DEFAULT_LLAMA_THREADS),
        n_gpu_layers=int(extra.get("n_gpu_layers", DEFAULT_LLAMA_NGL)),
        always_on=bool(extra.get("always_on", True)),
        ttl_seconds=extra.get("ttl_seconds") or (None if extra.get("always_on", True) else LLAMA_SWAP_TTL),
        extra_args=tuple(extra.get("llama_extra_args") or ()),
    )


def build_serve_specs(registry=None, *, only=None):
    registry = registry if registry is not None else get_model_registry()
    only = set(make_list(only)) if only else None
    specs = {}
    for key, cfg in registry.items():
        if only and key not in only:
            continue
        spec = serve_spec_for(key, cfg=cfg)
        specs[key] = spec
    return specs


def _assert_no_port_collisions(specs):
    seen = {}
    for spec in specs:
        seen.setdefault(spec.port, []).append(spec.model_key)
    clashes = {p: keys for p, keys in seen.items() if len(keys) > 1}
    if clashes:
        detail = "; ".join(f"{p} -> {keys}" for p, keys in sorted(clashes.items()))
        raise ValueError(f"systemd port collision: {detail}")


# --------------------------------------------------------------------------- #
# plan: a queue of writes + commands, executed only when you say so           #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class WriteFile:
    path: str
    content: str

    def describe(self) -> str:
        return f"write {self.path} ({len(self.content)} bytes)"


@dataclass(frozen=True)
class RunCmd:
    argv: Tuple[str, ...]

    def describe(self) -> str:
        return "$ " + " ".join(self.argv)


@dataclass(frozen=True)
class ServePlan:
    steps: Tuple[object, ...] = ()

    def __add__(self, other):
        return ServePlan(self.steps + other.steps)

    def describe(self):
        return [s.describe() for s in self.steps]


def apply_plan(plan, *, run=None, write=None):
    """Drain a ServePlan in order.

    Dry run (run=write=None) returns plan.describe() — hand that to the console.
    To execute, pass callables:
        run(argv)          e.g. partial(subprocess.run, check=True)
        write(path, text)  e.g. local writer, or an SSH writer for a peer node
    Local vs remote is entirely the caller's run/write — the plan is host-agnostic.
    """
    if run is None and write is None:
        return plan.describe()
    results = []
    for step in plan.steps:
        if isinstance(step, WriteFile):
            if write is None:
                raise RuntimeError("plan has file writes but no write() was provided")
            results.append(write(step.path, step.content))
        else:
            if run is None:
                raise RuntimeError("plan has commands but no run() was provided")
            results.append(run(list(step.argv)))
    return results


# --------------------------------------------------------------------------- #
# driver registry                                                             #
# --------------------------------------------------------------------------- #

_SERVE_DRIVERS = {}


def register_serve_driver(mode):
    def deco(cls):
        if mode in _SERVE_DRIVERS:
            raise KeyError(f"serve driver for {mode} already registered")
        _SERVE_DRIVERS[mode] = cls()          # stateless singleton
        return cls
    return deco


def get_serve_driver(mode):
    if mode not in _SERVE_DRIVERS:
        raise KeyError(f"no serve driver for mode={mode!r}; have {sorted(_SERVE_DRIVERS)}")
    return _SERVE_DRIVERS[mode]


# ---- systemd: one unit per model, always-on -------------------------------

@register_serve_driver(ServeMode.SYSTEMD)
class SystemdDriver:
    name = "systemd"

    def endpoint(self, spec):
        return f"http://{spec.host}:{spec.port}"

    def model_name(self, spec):
        return spec.swap_name

    def render_unit(self, spec):
        exec_start = " \\\n  ".join((
            spec.server_bin,
            f"-m {spec.model_file}",
            f"--host {spec.host}",
            f"--port {spec.port}",
            f"-c {spec.ctx_size}",
            f"-t {spec.threads}",
            # GPU offload — without this llama-server runs CPU-only.
            f"--n-gpu-layers {spec.n_gpu_layers}",
            *spec.extra_args,
        ))
        return "\n".join((
            "[Unit]",
            f"Description=llama.cpp server for {spec.model_key}",
            "After=network.target",
            # Restart-loop limiter lives in [Unit] (systemd ignores it in
            # [Service] — 'Unknown key name StartLimitIntervalSec in section
            # Service'). Caps thrash on a bad GGUF.
            "StartLimitIntervalSec=120",
            "StartLimitBurst=5",
            "",
            "[Service]",
            "Type=simple",
            f"User={spec.user}",
            f"Group={spec.group}",
            f"WorkingDirectory={spec.working_directory}",
            f"ExecStart={exec_start}",
            "Restart=always",
            "RestartSec=5",
            "TimeoutStartSec=300",
            "TimeoutStopSec=30",
            "",
            "[Install]",
            "WantedBy=multi-user.target",
            "",
        ))

    def install_plan(self, specs):
        steps = []
        for spec in specs:
            path = os.path.join(SYSTEMD_UNIT_DIR, spec.unit_name + ".service")
            steps.append(WriteFile(path, self.render_unit(spec)))
        steps.append(RunCmd(("systemctl", "daemon-reload")))
        for spec in specs:
            unit = spec.unit_name + ".service"
            steps.append(RunCmd(("systemctl", "enable", "--now" if spec.always_on else unit, unit)
                                if spec.always_on else ("systemctl", "enable", unit)))
        return ServePlan(tuple(steps))

    def start_plan(self, spec):
        return ServePlan((RunCmd(("systemctl", "start", spec.unit_name + ".service")),))

    def stop_plan(self, spec):
        return ServePlan((RunCmd(("systemctl", "stop", spec.unit_name + ".service")),))

    def status_plan(self, spec):
        return ServePlan((RunCmd(("systemctl", "is-active", spec.unit_name + ".service")),))


# ---- swap: one shared llama-swap proxy, on-demand -------------------------

@register_serve_driver(ServeMode.SWAP)
class SwapDriver:
    name = "swap"

    def endpoint(self, spec):
        return f"http://{spec.host}:{spec.port}"          # the proxy, same for all

    def model_name(self, spec):
        return spec.swap_name                             # request "model" field

    def render_config(self, specs):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("swap mode needs PyYAML: pip install pyyaml") from exc

        models = {}
        members = []
        for spec in specs:
            slug = spec.swap_name
            cmd = " ".join((
                spec.server_bin, "-m", spec.model_file,
                "--host", "127.0.0.1", "--port", "${PORT}",
                "-c", str(spec.ctx_size), "-t", str(spec.threads),
                "--n-gpu-layers", str(spec.n_gpu_layers),
                *spec.extra_args,
            ))
            entry = {"cmd": cmd}
            if spec.always_on:
                members.append(slug)
            elif spec.ttl_seconds:
                entry["ttl"] = spec.ttl_seconds
            models[slug] = entry

        cfg = {"models": models}
        if members:
            # always-on models share a group so they coexist instead of evicting
            # each other. NOTE: verify the coexistence flag names against your
            # llama-swap version — group schema has shifted across releases.
            cfg["groups"] = {"persistent": {"swap": False, "exclusive": False, "members": members}}
        return yaml.safe_dump(cfg, sort_keys=False)

    def install_plan(self, specs):
        steps = [
            WriteFile(LLAMA_SWAP_CONFIG, self.render_config(specs)),
            RunCmd(("systemctl", "restart", LLAMA_SWAP_UNIT)),
        ]
        return ServePlan(tuple(steps))

    def start_plan(self, spec):
        # on-demand: the proxy loads on first request. Optional warmup hit.
        return ServePlan(())

    def stop_plan(self, spec):
        url = f"http://{spec.host}:{spec.port}/unload?model={spec.swap_name}"
        return ServePlan((RunCmd(("curl", "-s", "-X", "POST", url)),))

    def status_plan(self, spec):
        return ServePlan((RunCmd(("curl", "-s", f"http://{spec.host}:{spec.port}/running")),))


# ---- off: in-process only, nothing to install -----------------------------

@register_serve_driver(ServeMode.OFF)
class OffDriver:
    name = "off"

    def endpoint(self, spec):
        return None                       # runner falls back to in-process

    def model_name(self, spec):
        return spec.swap_name

    def install_plan(self, specs):
        return ServePlan(())

    def start_plan(self, spec):
        return ServePlan(())

    def stop_plan(self, spec):
        return ServePlan(())

    def status_plan(self, spec):
        return ServePlan(())


# --------------------------------------------------------------------------- #
# top-level API — what the runner and the console call                        #
# --------------------------------------------------------------------------- #

def serve_endpoint(model_key) -> Optional[str]:
    """Base URL the HTTP runner should hit, or None for in-process."""
    spec = serve_spec_for(model_key)
    return get_serve_driver(spec.mode).endpoint(spec)


def serve_model_name(model_key) -> str:
    """Value to put in the request 'model' field (matters for swap routing)."""
    spec = serve_spec_for(model_key)
    return get_serve_driver(spec.mode).model_name(spec)


def install_serving(*, only=None, registry=None) -> ServePlan:
    """One plan that stands up everything. Batches by mode so all swap models
    land in one config write, each systemd model in its own unit."""
    specs = build_serve_specs(registry=registry, only=only)
    # Only the units we're about to write must not clash on a port.
    _assert_no_port_collisions([s for s in specs.values() if s.mode is ServeMode.SYSTEMD])
    by_mode = {}
    for spec in specs.values():
        by_mode.setdefault(spec.mode, []).append(spec)

    plan = ServePlan(())
    for mode, mode_specs in by_mode.items():
        plan = plan + get_serve_driver(mode).install_plan(mode_specs)
    return plan


def start_serving(model_key) -> ServePlan:
    spec = serve_spec_for(model_key)
    return get_serve_driver(spec.mode).start_plan(spec)


def stop_serving(model_key) -> ServePlan:
    spec = serve_spec_for(model_key)
    return get_serve_driver(spec.mode).stop_plan(spec)


def serving_overview(registry=None):
    """Rows for the console: what each model's serving looks like (no side effects)."""
    specs = build_serve_specs(registry=registry)
    rows = []
    for key, spec in specs.items():
        driver = get_serve_driver(spec.mode)
        rows.append(spec_row(spec, driver))
    return rows


def spec_row(spec, driver=None) -> dict:
    """One serving row for the console — the editable knobs + resolved endpoint."""
    driver = driver or get_serve_driver(spec.mode)
    return {
        "key": spec.model_key,
        "mode": spec.mode.value,
        "always_on": spec.always_on,
        "endpoint": driver.endpoint(spec),
        "model_name": driver.model_name(spec),
        "port": spec.port,
        "n_gpu_layers": spec.n_gpu_layers,
        "threads": spec.threads,
        "ctx_size": spec.ctx_size,
        "ttl_seconds": spec.ttl_seconds,
    }
