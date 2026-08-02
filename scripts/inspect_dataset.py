#!/usr/bin/env python3
"""Inspect a LeRobot dataset without reimplementing its dataset reader.

For v3 datasets, this uses ``LeRobotDataset``.  A v2.1 dataset is rejected by
LeRobot 0.4.1; in that case the script downloads only official metadata to
explain the conversion required, rather than pretending to load its frames.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from so101_sorting.baseline import (
    DATASET_SOURCE_REPO_ID,
    REQUIRED_CAMERA_KEYS,
    REQUIRED_VECTOR_KEYS,
    summarize_episode_lengths,
    validate_pick_orange_metadata,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id", default=DATASET_SOURCE_REPO_ID, help="Hugging Face dataset repository."
    )
    parser.add_argument(
        "--episode-index", type=int, default=0, help="Episode used for the contact sheet."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dataset_inspection"),
        help="Report directory.",
    )
    parser.add_argument(
        "--root", type=Path, help="Optional local LeRobotDataset root (v3 after conversion)."
    )
    parser.add_argument(
        "--no-download-videos",
        action="store_true",
        help="Do not decode camera frames for the contact sheet.",
    )
    parser.add_argument(
        "--video-backend", default="pyav", help="LeRobot video backend (default: pyav)."
    )
    return parser.parse_args(argv)


def _write_report(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "dataset_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(f"Report written to {report_path}")


def _download_v2_metadata(
    repo_id: str, root: Path | None = None
) -> tuple[dict[str, Any], list[int]]:
    if root is not None:
        info_path = root / "meta/info.json"
        episodes_path = root / "meta/episodes.jsonl"
        if not info_path.is_file():
            raise RuntimeError(f"Local dataset metadata not found: {info_path}")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        lengths: list[int] = []
        if episodes_path.is_file():
            for line in episodes_path.read_text(encoding="utf-8").splitlines():
                if line.strip() and isinstance((episode := json.loads(line)).get("length"), int):
                    lengths.append(episode["length"])
        return info, lengths
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to inspect metadata before LeRobot can open this dataset. "
            "Install the pinned LeRobot environment first."
        ) from exc

    info_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="meta/info.json")
    info = json.loads(Path(info_path).read_text(encoding="utf-8"))
    lengths: list[int] = []
    try:
        episodes_path = hf_hub_download(
            repo_id=repo_id, repo_type="dataset", filename="meta/episodes.jsonl"
        )
        for line in Path(episodes_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                episode = json.loads(line)
                if isinstance(episode.get("length"), int):
                    lengths.append(episode["length"])
    except (
        Exception
    ) as exc:  # Metadata is supplementary; preserve the primary compatibility result.
        print(f"Warning: unable to read meta/episodes.jsonl: {exc}", file=sys.stderr)
    return info, lengths


def _feature_summary(features: dict[str, Any]) -> dict[str, Any]:
    return {
        key: {
            name: value.get(name)
            for name in ("dtype", "shape", "names", "codec", "fps")
            if isinstance(value, dict) and name in value
        }
        for key, value in features.items()
    }


def _array_stats(values: Any) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.number):
        return {"dtype": str(array.dtype), "shape": list(array.shape), "numeric": False}
    finite = np.isfinite(array)
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "numeric": True,
        "nan_count": int(np.isnan(array).sum()),
        "inf_count": int(np.isinf(array).sum()),
        "min": float(np.min(array[finite])) if finite.any() else None,
        "max": float(np.max(array[finite])) if finite.any() else None,
    }


def _save_contact_sheet(sample: dict[str, Any], output_path: Path) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to create the camera contact sheet.") from exc
    import numpy as np

    images: list[Image.Image] = []
    for key in REQUIRED_CAMERA_KEYS:
        frame = sample[key]
        if hasattr(frame, "detach"):
            frame = frame.detach().cpu().numpy()
        array = np.asarray(frame)
        if array.ndim == 3 and array.shape[0] in {1, 3, 4}:  # CHW from LeRobot transforms.
            array = np.moveaxis(array, 0, -1)
        if array.dtype.kind == "f":
            array = np.clip(array * (255 if array.max(initial=0) <= 1 else 1), 0, 255).astype(
                np.uint8
            )
        images.append(Image.fromarray(array).convert("RGB"))

    width = max(image.width for image in images)
    height = sum(image.height for image in images)
    sheet = Image.new("RGB", (width, height), "black")
    y_offset = 0
    for image in images:
        sheet.paste(image, (0, y_offset))
        y_offset += image.height
    sheet.save(output_path)


def _inspect_v3(args: argparse.Namespace) -> int:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        print(
            "LeRobot is not importable. Install the project stack with LeRobot 0.4.1, then retry. "
            f"Original error: {exc}",
            file=sys.stderr,
        )
        return 3

    try:
        dataset = LeRobotDataset(
            args.repo_id,
            root=args.root,
            download_videos=False,
            video_backend=args.video_backend,
        )
    except Exception as exc:
        message = str(exc)
        if "v2.1" in message or "BackwardCompatibility" in type(exc).__name__:
            return _report_v2_incompatibility(args, message)
        print(f"Unable to load LeRobotDataset '{args.repo_id}': {message}", file=sys.stderr)
        return 3

    info = dict(dataset.meta.info)
    validation = validate_pick_orange_metadata(info)
    if args.episode_index < 0 or args.episode_index >= dataset.meta.total_episodes:
        print(
            f"Episode {args.episode_index} is out of range [0, {dataset.meta.total_episodes - 1}].",
            file=sys.stderr,
        )
        return 2
    episode_metadata = dataset.meta.episodes
    if "length" in episode_metadata.column_names:
        lengths = [int(length) for length in episode_metadata["length"]]
    else:
        lengths = [
            int(episode["dataset_to_index"] - episode["dataset_from_index"])
            for episode in episode_metadata
        ]
    report: dict[str, Any] = {
        "repo_id": args.repo_id,
        "codebase_version": info.get("codebase_version"),
        "total_episodes": dataset.meta.total_episodes,
        "total_frames": dataset.meta.total_frames,
        "fps": dataset.fps,
        "features": _feature_summary(dict(dataset.meta.features)),
        "camera_keys": list(dataset.meta.camera_keys),
        "validation_errors": list(validation.errors),
        "validation_warnings": list(validation.warnings),
        "episode_durations": summarize_episode_lengths(lengths, dataset.fps),
        "numeric_fields": {},
    }
    for key in REQUIRED_VECTOR_KEYS:
        report["numeric_fields"][key] = _array_stats(dataset.hf_dataset[key])

    if not args.no_download_videos:
        try:
            visual_dataset = LeRobotDataset(
                args.repo_id,
                root=args.root,
                episodes=[args.episode_index],
                download_videos=True,
                video_backend=args.video_backend,
            )
            sample = visual_dataset[0]
            contact_sheet = args.output_dir / f"episode_{args.episode_index:03d}_cameras.png"
            args.output_dir.mkdir(parents=True, exist_ok=True)
            _save_contact_sheet(sample, contact_sheet)
            report["contact_sheet"] = str(contact_sheet)
        except Exception as exc:
            report["contact_sheet_error"] = str(exc)
            print(f"Camera visualization failed: {exc}", file=sys.stderr)
    _write_report(args.output_dir, report)
    print(json.dumps(report, indent=2, default=str))
    return 0 if validation.is_compatible_schema else 2


def _report_v2_incompatibility(args: argparse.Namespace, load_error: str | None = None) -> int:
    try:
        info, lengths = _download_v2_metadata(args.repo_id, args.root)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    validation = validate_pick_orange_metadata(info)
    report: dict[str, Any] = {
        "repo_id": args.repo_id,
        "codebase_version": info.get("codebase_version"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "fps": info.get("fps"),
        "features": _feature_summary(dict(info.get("features", {}))),
        "camera_keys_present": {
            key: key in info.get("features", {}) for key in REQUIRED_CAMERA_KEYS
        },
        "validation_errors": list(validation.errors),
        "validation_warnings": list(validation.warnings),
        "load_error": load_error,
        "required_conversion": (
            "python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 "
            f"--repo-id={args.repo_id} --root=<local-cache-parent> --push-to-hub=false"
        ),
    }
    if lengths and info.get("fps"):
        report["episode_durations"] = summarize_episode_lengths(lengths, float(info["fps"]))
    _write_report(args.output_dir, report)
    print(
        f"Dataset '{args.repo_id}' is LeRobotDataset {info.get('codebase_version')}, which LeRobot 0.4.1 "
        "does not load directly. Convert it to v3 before training or visualizing frames.",
        file=sys.stderr,
    )
    print(report["required_conversion"], file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Read only published metadata first. This lets a pre-installation run
    # identify the v2.1 blocker without downloading frames or videos.
    try:
        info, _ = _download_v2_metadata(args.repo_id, args.root)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    if info.get("codebase_version") != "v3.0":
        return _report_v2_incompatibility(
            args, "LeRobotDataset v3 was not attempted: metadata is not v3.0."
        )
    return _inspect_v3(args)


if __name__ == "__main__":
    raise SystemExit(main())
