#!/usr/bin/env python3
"""Export a compact visual/control summary for one current-stack recorded episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.validate_recorded_dataset import _scalar, episode_columns


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/current_stack"))
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dataset_visualization"))
    return parser.parse_args(argv)


def _write_svg(values: list[list[float]], path: Path, title: str) -> None:
    """Write a dependency-free six-joint line plot."""

    width, height, padding = 900, 400, 45
    flattened = [value for frame in values for value in frame]
    low, high = min(flattened), max(flattened)
    span = max(high - low, 1e-9)
    frame_span = max(len(values) - 1, 1)
    colors = ["#dc2626", "#ea580c", "#ca8a04", "#16a34a", "#2563eb", "#7c3aed"]
    lines = []
    for joint in range(6):
        points = " ".join(
            f"{padding + frame / frame_span * (width - 2 * padding):.2f},"
            f"{height - padding - (values[frame][joint] - low) / span * (height - 2 * padding):.2f}"
            for frame in range(len(values))
        )
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{colors[joint]}" stroke-width="1.5"/>'
        )
    path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                f'<text x="{padding}" y="24">{title}</text>',
                f'<path d="M {padding} {padding} V {height - padding} H {width - padding}" stroke="black" fill="none"/>',
                *lines,
                f'<text x="{padding}" y="{height - 12}">frames: {len(values)}</text>',
                "</svg>",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_ppm(image: object, path: Path) -> None:
    """Store RGB observations without adding a Pillow/matplotlib runtime dependency."""

    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.moveaxis(array, 0, -1)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected RGB image HWC/CHW, got {array.shape}")
    if array.dtype.kind == "f":
        array = np.clip(array * 255 if array.max(initial=0) <= 1 else array, 0, 255)
    array = np.ascontiguousarray(array.astype(np.uint8))
    height, width, _ = array.shape
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode() + array.tobytes())


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print("Visualization requires LeRobot 0.4.1 and a local LeRobotDataset v3.")
        return 3
    try:
        dataset = LeRobotDataset(args.repo_id, root=args.dataset_root / args.repo_id)
        episode = episode_columns(
            dataset, args.episode_index, ["observation.state", "action", "timestamp"]
        )
    except (IndexError, KeyError, OSError, ValueError) as exc:
        print(f"Cannot load episode {args.episode_index}: {exc}")
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    states = [[float(value) for value in frame] for frame in episode["observation.state"]]
    actions = [[float(value) for value in frame] for frame in episode["action"]]
    _write_svg(states, args.output_dir / "joint_state.svg", "SO-101 measured joint state (radians)")
    _write_svg(actions, args.output_dir / "joint_action.svg", "SO-101 IK joint target (radians)")
    metadata = dataset.meta.episodes[args.episode_index]
    start = _scalar(metadata["dataset_from_index"])
    end = _scalar(metadata["dataset_to_index"])
    camera_exports: list[str] = []
    for frame_index in sorted({start, start + max((end - start - 1) // 2, 0), end - 1}):
        frame = dataset[frame_index]
        for camera in ("observation.images.front", "observation.images.wrist"):
            path = (
                args.output_dir
                / f"{camera.rsplit('.', maxsplit=1)[-1]}_{frame_index - start:04d}.ppm"
            )
            _write_ppm(frame[camera], path)
            camera_exports.append(str(path))
    summary = {
        "repo_id": args.repo_id,
        "episode_index": args.episode_index,
        "frames": len(actions),
        "duration_seconds": float(episode["timestamp"][-1]) if actions else 0.0,
        "camera_features": ["observation.images.front", "observation.images.wrist"],
        "camera_exports": camera_exports,
        "success": "The recorder exports only accepted episodes; inspect the collection sidecar for the protocol.",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print("Exported front/wrist RGB frames (PPM) and joint/action traces (SVG).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
