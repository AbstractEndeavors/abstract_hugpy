"""Phone-brick: distributed PPE detection across a chain of phone workers.

A small mechanic for running ONNX YOLO PPE detection on a fleet of cheap Android
phones (Termux) and collating their verdicts by plurality consensus. Two halves:

* the **worker** (:mod:`.worker`) — runs on each phone, serving a ``yolo <url>``
  HTTP verb backed by :class:`.detector.OnnxYoloDetector`;
* the **orchestrator** (:mod:`.orchestrator`) — runs on a control box, fanning
  one image across the chain and writing an annotated, consensus-labelled result.

The two share a single wire :mod:`.protocol`, so the worker's output format and
the orchestrator's parser can never drift apart.

    python -m abstract_hugpy.phone_brick worker
    python -m abstract_hugpy.phone_brick orchestrate --image img.jpg \\
        --file-server http://192.168.0.26:8088/chain-outputs/ \\
        --phones red:192.168.0.32:5002:#f85149,blue:192.168.0.70:5003:#58a6ff
"""
from .schemas import (
    ChainConfig, ChainResult, Detection, DetectorConfig, PhaseResult,
    PhoneSpec, WorkerConfig,
)
from .detector import OnnxYoloDetector
from .protocol import format_detections, parse_detections
from .consensus import plurality_consensus
from .client import WorkerClient
from .rendering import draw_detections
from .orchestrator import ChainOrchestrator
from .worker import Worker, build_app, config_from_env

__all__ = [
    "ChainConfig", "ChainResult", "Detection", "DetectorConfig",
    "PhaseResult", "PhoneSpec", "WorkerConfig",
    "OnnxYoloDetector",
    "format_detections", "parse_detections",
    "plurality_consensus",
    "WorkerClient",
    "draw_detections",
    "ChainOrchestrator",
    "Worker", "build_app", "config_from_env",
]
