# imports/apis/systemd_units.py
"""
Generate systemd units for the llama.cpp-served models, straight from
MODEL_REGISTRY. The registry is the source of truth; a unit is just a
rendering of one ModelConfig whose framework is "llama_cpp".

Wiring (build dir / user / group / unit dir) is explicit env, never sniffed
from disk. Path resolution reuses resolve_model_source / get_gguf_file so this
module doesn't re-decide where a GGUF lives. Rendering is pure; writing files
and the systemctl steps are separate, inspectable calls — importing this does
nothing to the host.

os.path throughout, no pathlib.
"""


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

# model_max_length can be a tokenizer sentinel (~1e30) or huge (131072). Never
# feed that to -c. Cap to this unless cfg.extra["llama_ctx"] says otherwise.
DEFAULT_LLAMA_CTX = int(get_env_value("DEFAULT_LLAMA_CTX") or 4096)
DEFAULT_LLAMA_THREADS = int(get_env_value("DEFAULT_LLAMA_THREADS") or 6)


# --------------------------------------------------------------------------- #
# small resolvers — each reuses existing package logic, adds nothing new       #
# --------------------------------------------------------------------------- #

def _unit_slug(value):
    """systemd-safe bare unit name (no '.service'): a-z0-9._- only."""
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-._")
    return value or "model"


def _bare_host(value):
    """llama-server --host wants a bare host, not a scheme. LLAMA_HOST is a URL."""
    value = value or LLAMA_HOST
    if "://" in value:
        value = value.split("://", 1)[1]
    return value.split("/", 1)[0] or "127.0.0.1"


def _ctx_for(cfg, model_key):
    override = (getattr(cfg, "extra", {}) or {}).get("llama_ctx")
    if override:
        return int(override)
    mml = getattr(cfg, "model_max_length", None) or DEFAULT_LLAMA_CTX
    capped = min(int(mml), DEFAULT_LLAMA_CTX)
    if capped < int(mml):
        logger.info(
            f"{model_key}: capping -c {int(mml)} -> {capped} "
            f"(set extra['llama_ctx'] to override)"
        )
    return capped


def _model_file_for(model_key, cfg, require_exists=True):
    """Absolute GGUF path. resolve_model_source returns the file when the model
    is downloaded+complete, else the hub_id — so verify it's actually a file
    before trusting it, then fall back to the explicit folder/filename."""
    try:
        source = resolve_model_source(model_key)
        if source and os.path.isfile(source):
            return source
    except (FileNotFoundError, KeyError) as exc:
        logger.info(f"{model_key}: resolve_model_source unavailable ({exc})")

    candidate = None
    if getattr(cfg, "filename", None):
        candidate = os.path.join(get_model_path(model_key), cfg.filename)
        if os.path.isfile(candidate):
            return candidate

    if require_exists:
        raise FileNotFoundError(
            f"{model_key}: no local GGUF found "
            f"(filename={getattr(cfg, 'filename', None)!r}). "
            f"Run ensure_model({model_key!r}) first, or build with "
            f"require_exists=False to emit the unit anyway."
        )
    return candidate or os.path.join(get_model_path(model_key), cfg.filename or "")


# --------------------------------------------------------------------------- #
# schema: a validated, frozen unit spec                                       #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LlamaServiceSpec:
    model_key: str
    description: str
    model_file: str                 # absolute .gguf path
    port: int
    host: str = "127.0.0.1"
    ctx_size: int = DEFAULT_LLAMA_CTX
    threads: int = DEFAULT_LLAMA_THREADS
    user: str = LLAMA_SERVICE_USER
    group: str = LLAMA_SERVICE_GROUP
    working_directory: str = LLAMA_CPP_DIR
    server_bin: str = LLAMA_SERVER_BIN
    restart_sec: int = 5
    timeout_start_sec: int = 300
    timeout_stop_sec: int = 30

    def __post_init__(self):
        if not os.path.isabs(self.server_bin):
            raise ValueError(f"{self.model_key}: server_bin must be absolute: {self.server_bin!r}")
        if not os.path.isabs(self.working_directory):
            raise ValueError(f"{self.model_key}: working_directory must be absolute")
        if not os.path.isabs(self.model_file):
            raise ValueError(f"{self.model_key}: model_file must be absolute: {self.model_file!r}")
        if not 0 < self.port < 65536:
            raise ValueError(f"{self.model_key}: port out of range: {self.port}")
        if self.threads < 1:
            raise ValueError(f"{self.model_key}: threads must be >= 1")
        if self.ctx_size < 1:
            raise ValueError(f"{self.model_key}: ctx_size must be >= 1")

    @property
    def unit_name(self):
        return _unit_slug(f"{LLAMA_UNIT_PREFIX}-{self.model_key}")


def spec_from_model_config(cfg, *, model_key=None, require_exists=True):
    """ModelConfig -> LlamaServiceSpec. Raises for non-llama_cpp or portless."""
    model_key = model_key or getattr(cfg, "model_key", None) or getattr(cfg, "name", None)
    if getattr(cfg, "framework", None) != "llama_cpp":
        raise ValueError(f"{model_key}: framework is {getattr(cfg, 'framework', None)!r}, not llama_cpp")
    if not getattr(cfg, "port", None):
        raise ValueError(f"{model_key}: no port set; a served model needs one")

    return LlamaServiceSpec(
        model_key=model_key,
        description=f"llama.cpp server for {getattr(cfg, 'name', None) or model_key}",
        model_file=_model_file_for(model_key, cfg, require_exists=require_exists),
        port=int(cfg.port),
        host=_bare_host(getattr(cfg, "host", None)),
        ctx_size=_ctx_for(cfg, model_key),
    )


# --------------------------------------------------------------------------- #
# rendering: LlamaServiceSpec -> unit text                                    #
# --------------------------------------------------------------------------- #

def render_unit(spec):
    exec_start = " \\\n  ".join((
        spec.server_bin,
        f"-m {spec.model_file}",
        f"--host {spec.host}",
        f"--port {spec.port}",
        f"-c {spec.ctx_size}",
        f"-t {spec.threads}",
    ))
    return "\n".join((
        "[Unit]",
        f"Description={spec.description}",
        "After=network.target",
        "",
        "[Service]",
        "Type=simple",
        f"User={spec.user}",
        f"Group={spec.group}",
        f"WorkingDirectory={spec.working_directory}",
        f"ExecStart={exec_start}",
        "Restart=always",
        f"RestartSec={spec.restart_sec}",
        f"TimeoutStartSec={spec.timeout_start_sec}",
        f"TimeoutStopSec={spec.timeout_stop_sec}",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
        "",
    ))


# --------------------------------------------------------------------------- #
# registry view: the llama_cpp slice of MODEL_REGISTRY as specs               #
# --------------------------------------------------------------------------- #

def _assert_no_port_collisions(specs):
    seen = {}
    for spec in specs.values():
        seen.setdefault(spec.port, []).append(spec.model_key)
    clashes = {port: keys for port, keys in seen.items() if len(keys) > 1}
    if clashes:
        detail = "; ".join(f"{port} -> {keys}" for port, keys in sorted(clashes.items()))
        raise ValueError(f"port collision across llama_cpp models: {detail}")


def build_llama_specs(model_registry=None, *, require_exists=True, only=None):
    """Return {unit_name: LlamaServiceSpec} for every llama_cpp model with a port.

    Portless or undownloaded models (when require_exists) are skipped with a log
    line, not a crash, so one bad row doesn't sink the whole generation.
    """
    model_registry = model_registry if model_registry is not None else get_model_registry()
    only = set(make_list(only)) if only else None

    specs = {}
    for key, cfg in model_registry.items():
        if getattr(cfg, "framework", None) != "llama_cpp":
            continue
        if only and key not in only:
            continue
        if not getattr(cfg, "port", None):
            logger.info(f"{key}: skipped, no port")
            continue
        try:
            spec = spec_from_model_config(cfg, model_key=key, require_exists=require_exists)
        except (FileNotFoundError, ValueError) as exc:
            logger.info(f"{key}: skipped, {exc}")
            continue
        specs[spec.unit_name] = spec

    _assert_no_port_collisions(specs)
    return specs


# --------------------------------------------------------------------------- #
# side effects: write files, plan systemctl, drain the plan                   #
# --------------------------------------------------------------------------- #

def write_unit(spec, unit_dir=None):
    unit_dir = unit_dir or SYSTEMD_UNIT_DIR
    os.makedirs(unit_dir, exist_ok=True)
    path = os.path.join(unit_dir, spec.unit_name + ".service")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_unit(spec))
    return path


def write_all(specs, unit_dir=None):
    return {name: write_unit(spec, unit_dir) for name, spec in specs.items()}


def systemctl_plan(specs, enable=True, start=True):
    """Ordered command queue to apply these units. Inspect/diff it before you
    run it; nothing executes here."""
    plan = [["systemctl", "daemon-reload"]]
    for spec in specs.values():
        unit = spec.unit_name + ".service"
        if enable and start:
            plan.append(["systemctl", "enable", "--now", unit])
        elif enable:
            plan.append(["systemctl", "enable", unit])
        elif start:
            plan.append(["systemctl", "start", unit])
    return plan


def apply_plan(plan, runner=None):
    """Drain the queue. runner=None is a dry run (returns the commands). For
    real use: apply_plan(plan, runner=partial(subprocess.run, check=True))."""
    if runner is None:
        return list(plan)
    return [runner(cmd) for cmd in plan]


def generate_llama_units(unit_dir=None, *, require_exists=True, only=None, write=True):
    """One call: build specs from the registry, optionally write them, and
    return (specs, written_paths, plan). Mirrors ensure_models()'s shape."""
    specs = build_llama_specs(require_exists=require_exists, only=only)
    written = write_all(specs, unit_dir) if write else {}
    return specs, written, systemctl_plan(specs)


# --------------------------------------------------------------------------- #
# CLI — mirrors download_dict_models / ensure_models ergonomics               #
# --------------------------------------------------------------------------- #

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate systemd units for llama.cpp-served models from MODEL_REGISTRY."
    )
    parser.add_argument("--unit-dir", default=SYSTEMD_UNIT_DIR, help="where .service files go")
    parser.add_argument("--only", nargs="*", default=None, help="limit to these model keys")
    parser.add_argument("--allow-missing", action="store_true",
                        help="emit units even if the GGUF isn't downloaded yet")
    parser.add_argument("--no-write", action="store_true",
                        help="print units + plan only, write nothing")
    args = parser.parse_args()

    specs = build_llama_specs(require_exists=not args.allow_missing, only=args.only)
    if not specs:
        print("# no llama_cpp units to generate")
        return

    for spec in specs.values():
        print(f"# === {spec.unit_name}.service ===")
        print(render_unit(spec))

    if not args.no_write:
        written = write_all(specs, args.unit_dir)
        print("# wrote:")
        for path in written.values():
            print(f"#   {path}")

    print("# apply plan:")
    for cmd in systemctl_plan(specs):
        print(" ".join(cmd))


if __name__ == "__main__":
    main()
