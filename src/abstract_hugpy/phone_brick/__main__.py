"""CLI for the phone-brick mechanic.

    python -m abstract_hugpy.phone_brick worker
    python -m abstract_hugpy.phone_brick orchestrate --image img.jpg \\
        --file-server http://192.168.0.26:8088/chain-outputs/ \\
        --phones red:192.168.0.32:5002:#f85149,blue:192.168.0.70:5003:#58a6ff
"""
from __future__ import annotations

import argparse
import sys

from .schemas import ChainConfig, PhoneSpec


def _parse_phones(spec: str) -> list[PhoneSpec]:
    """Parse ``name:host[:port[:#hex]]`` comma-separated phone specs."""
    phones: list[PhoneSpec] = []
    for token in (s.strip() for s in spec.split(",") if s.strip()):
        parts = token.split(":")
        if len(parts) < 2:
            raise ValueError(
                f"bad phone spec {token!r}; expected name:host[:port[:#hex]]")
        name, host = parts[0], parts[1]
        port = int(parts[2]) if len(parts) >= 3 and parts[2] else 5002
        color = parts[3] if len(parts) >= 4 and parts[3] else "#58a6ff"
        phones.append(PhoneSpec(name=name, host=host, port=port, color_hex=color))
    return phones


def _cmd_worker(args) -> int:
    from .worker import main as worker_main
    return worker_main()


def _cmd_orchestrate(args) -> int:
    phones = _parse_phones(args.phones)
    config = ChainConfig(
        phones=phones,
        file_server=args.file_server,
        push_timeout_s=args.push_timeout,
        drain_timeout_s=args.drain_timeout,
    )
    from .orchestrator import ChainOrchestrator

    orch = ChainOrchestrator(config)
    result = orch.run(args.image, args.output_dir)

    chain = " -> ".join(p.name for p in phones)
    print(f"chain: {chain}   image: {result.image}")
    for phase in result.phases:
        print(f"  [{phase.phone}] {phase.top_cls}_{phase.top_conf_pct} "
              f"{phase.consensus}  ({len(phase.detections)} det(s))")
    print(f"CHAIN COMPLETE: {result.output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="abstract_hugpy.phone_brick")
    sub = parser.add_subparsers(dest="command", required=True)

    w = sub.add_parser("worker", help="run a phone worker (config via env)")
    w.set_defaults(func=_cmd_worker)

    o = sub.add_parser("orchestrate", help="fan one image across a phone chain")
    o.add_argument("--image", required=True)
    o.add_argument("--output-dir", default="chain-outputs")
    o.add_argument("--file-server", required=True,
                   help="base URL the phones fetch the seeded image from")
    o.add_argument("--phones", required=True,
                   help="comma list of name:host[:port[:#hex]]")
    o.add_argument("--push-timeout", type=float, default=5.0)
    o.add_argument("--drain-timeout", type=float, default=60.0)
    o.set_defaults(func=_cmd_orchestrate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
