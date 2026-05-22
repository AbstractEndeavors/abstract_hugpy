from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FrameCandidate:
    timestamp: float
    reason: str
    segment_index: int
    text: str


def get_segment_frame_times(
    whisper_result: dict[str, Any],
    min_gap_seconds: float = 1.5,
    long_segment_seconds: float = 8.0,
) -> list[FrameCandidate]:
    candidates: list[FrameCandidate] = []
    last_timestamp = -math.inf

    for index, segment in enumerate(whisper_result.get("segments", [])):
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        text = str(segment.get("text", "")).strip()
        duration = max(0.0, end - start)

        if duration <= 0.0:
            continue

        if duration >= long_segment_seconds:
            timestamps = [
                (start + min(1.0, duration * 0.15), "segment_start_context"),
                ((start + end) / 2.0, "segment_midpoint"),
                (end - min(1.0, duration * 0.15), "segment_end_context"),
            ]
        else:
            timestamps = [((start + end) / 2.0, "segment_midpoint")]

        for timestamp, reason in timestamps:
            if timestamp - last_timestamp < min_gap_seconds:
                continue

            candidates.append(
                FrameCandidate(
                    timestamp=timestamp,
                    reason=reason,
                    segment_index=index,
                    text=text,
                )
            )
            last_timestamp = timestamp

    return candidates


def extract_frame_ffmpeg(
    video_path: str,
    timestamp: float,
    output_path: str,
    quality: int = 2,
) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        video_path,
        "-frames:v",
        "1",
        "-q:v",
        str(quality),
        output_path,
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed to extract frame.\n\n"
            f"Command:\n{' '.join(command)}\n\n"
            f"stderr:\n{result.stderr}"
        )

    return output_path


def extract_context_frames_from_whisper(
    video_path: str,
    whisper_result: dict[str, Any],
    output_dir: str,
    min_gap_seconds: float = 1.5,
    long_segment_seconds: float = 8.0,
) -> list[dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)

    candidates = get_segment_frame_times(
        whisper_result=whisper_result,
        min_gap_seconds=min_gap_seconds,
        long_segment_seconds=long_segment_seconds,
    )

    extracted: list[dict[str, Any]] = []

    for item_index, candidate in enumerate(candidates):
        frame_name = (
            f"frame_{item_index:04d}_"
            f"seg_{candidate.segment_index:04d}_"
            f"{candidate.timestamp:.3f}s.jpg"
        )
        frame_path = os.path.join(output_dir, frame_name)

        extract_frame_ffmpeg(
            video_path=video_path,
            timestamp=candidate.timestamp,
            output_path=frame_path,
        )

        extracted.append(
            {
                "frame_path": frame_path,
                "timestamp": candidate.timestamp,
                "reason": candidate.reason,
                "segment_index": candidate.segment_index,
                "text": candidate.text,
            }
        )

    frame_context_path = os.path.join(os.path.dirname(output_dir), "frame_context.json")
    with open(frame_context_path, "w", encoding="utf-8") as file:
        json.dump(extracted, file, indent=2, ensure_ascii=False)

    return extracted
