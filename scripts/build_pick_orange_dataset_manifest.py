#!/usr/bin/env python3
"""Create a machine-readable manifest from published PickOrange metadata only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="LightwheelAI/leisaac-pick-orange")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/compatibility/dataset_manifest.json")
    )
    return parser.parse_args(argv)


def _node(value: Any, source: str, notes: str = "") -> dict[str, Any]:
    return {"value": value, "status": "confirmed", "source": source, "notes": notes}


def build_manifest(info: dict[str, Any], repo_id: str) -> dict[str, Any]:
    """Map only directly published metadata into the comparison schema."""

    features = info["features"]
    action = features["action"]
    state = features["observation.state"]
    source = f"{repo_id}:meta/info.json"

    def camera(key: str) -> dict[str, Any]:
        feature = features[key]
        video_info = feature.get("video_info", feature.get("info", {}))
        return {
            "dataset_key": _node(key, source),
            "recorded_resolution": _node(feature["shape"], source, "Published as HWC."),
            "codec": _node(
                f"{video_info.get('video.codec')}/{video_info.get('video.pix_fmt')}", source
            ),
        }

    return {
        "dataset": {
            "format_version": _node(info["codebase_version"], source),
            "repository": _node(repo_id, source),
            "total_episodes": _node(info["total_episodes"], source),
            "total_frames": _node(info["total_frames"], source),
        },
        "robot": {"type": _node(info["robot_type"], source)},
        "actions": {
            "dimension": _node(action["shape"][0], source),
            "joint_order": _node(action["names"], source),
        },
        "state": {
            "dimension": _node(state["shape"][0], source),
            "joint_order": _node(state["names"], source),
        },
        "cameras": {
            "front": camera("observation.images.front"),
            "wrist": camera("observation.images.wrist"),
        },
        "simulation": {"dataset_frequency_hz": _node(info["fps"], source)},
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "Missing huggingface_hub. Run with the pinned LeRobot environment or install the lightweight client."
        )
        return 3
    try:
        path = hf_hub_download(args.repo_id, "meta/info.json", repo_type="dataset")
        info = json.loads(Path(path).read_text(encoding="utf-8"))
        manifest = build_manifest(info, args.repo_id)
    except Exception as exc:
        print(f"Cannot obtain dataset metadata: {exc}")
        return 3
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Dataset manifest written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
