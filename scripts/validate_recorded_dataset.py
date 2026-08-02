#!/usr/bin/env python3
"""Validate a locally recorded current-stack PickOrange LeRobotDataset v3."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import yaml

from so101_sorting.current_stack import environment_fingerprint, validate_episode_arrays

REQUIRED_FEATURES = {
    "observation.state",
    "action",
    "observation.images.front",
    "observation.images.wrist",
    "timestamp",
    "episode_index",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Local LeRobot dataset repository ID.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/current_stack"))
    parser.add_argument(
        "--frozen-config", type=Path, default=Path("configs/pick_orange_current_stack.yaml")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dataset_validation"))
    return parser.parse_args(argv)


def dataset_path(dataset_root: Path, repo_id: str) -> Path:
    return dataset_root / repo_id


def _result(status: str, message: str, *, blocking: bool = False) -> dict[str, Any]:
    return {"status": status, "message": message, "blocking": blocking}


def _read_sidecar(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def episode_columns(dataset: Any, episode_index: int, columns: list[str]) -> dict[str, Any]:
    """Read an episode through LeRobot's v3-backed dataset rather than its parquet files."""

    dataset._ensure_hf_dataset_loaded()  # LeRobot v0.4.1 lazy-load hook.
    metadata = dataset.meta.episodes[episode_index]
    start = _scalar(metadata["dataset_from_index"])
    end = _scalar(metadata["dataset_to_index"])
    return dataset.hf_dataset.select_columns(columns)[start:end]


def _scalar(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        value = value[0]
    return int(value)


def _as_list(values: Any) -> list[Any]:
    if hasattr(values, "detach"):
        values = values.detach().cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    return values if isinstance(values, list) else []


def validate_dataset(
    repo_id: str, dataset_root: Path, frozen_config: dict[str, Any]
) -> dict[str, Any]:  # noqa: C901
    """Validate structural invariants; imports LeRobot only after local files are resolved."""

    root = dataset_path(dataset_root, repo_id)
    findings: list[dict[str, Any]] = []
    sidecar_path = root / "meta" / "current_stack_manifest.json"
    if not root.is_dir():
        findings.append(_result("FAIL", f"Dataset directory does not exist: {root}", blocking=True))
        return _summary(repo_id, root, findings)
    if not sidecar_path.is_file():
        findings.append(
            _result("FAIL", f"Missing collection sidecar: {sidecar_path}", blocking=True)
        )
        return _summary(repo_id, root, findings)
    try:
        sidecar = _read_sidecar(sidecar_path)
    except (OSError, ValueError) as exc:
        findings.append(_result("FAIL", f"Unreadable collection sidecar: {exc}", blocking=True))
        return _summary(repo_id, root, findings)
    expected_fingerprint = environment_fingerprint(frozen_config)
    if sidecar.get("environment_fingerprint") != expected_fingerprint:
        findings.append(
            _result(
                "FAIL",
                "Dataset sidecar fingerprint differs from the frozen environment contract.",
                blocking=True,
            )
        )
    else:
        findings.append(_result("PASS", "Dataset sidecar fingerprint matches frozen environment."))
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        findings.append(
            _result("FAIL", "LeRobot 0.4.1 is required to validate recorded frames.", blocking=True)
        )
        return _summary(repo_id, root, findings, sidecar=sidecar)
    dataset = LeRobotDataset(repo_id, root=root)
    info = dataset.meta.info
    version = str(info.get("codebase_version", "unknown"))
    if not version.startswith("v3"):
        findings.append(
            _result("FAIL", f"Expected LeRobotDataset v3, got {version}.", blocking=True)
        )
    else:
        findings.append(_result("PASS", f"LeRobotDataset format: {version}."))
    features = set(dataset.features)
    missing = sorted(REQUIRED_FEATURES - features)
    if missing:
        findings.append(_result("FAIL", f"Missing required features: {missing}", blocking=True))
    else:
        findings.append(
            _result("PASS", "All required state, action, image and time features exist.")
        )
    expected_order = frozen_config["actions"]["order"]
    joint_order_mismatch = False
    for key in ("observation.state", "action"):
        feature = dataset.features.get(key, {})
        names = feature.get("names") if isinstance(feature, dict) else None
        if names != expected_order:
            joint_order_mismatch = True
            findings.append(
                _result(
                    "FAIL",
                    f"{key} joint order is {names!r}; expected {expected_order!r}.",
                    blocking=True,
                )
            )
    if not joint_order_mismatch:
        findings.append(_result("PASS", "State and action joint orders match the frozen contract."))
    fps = float(info.get("fps", 0.0))
    expected_fps = float(frozen_config["simulation"]["recording_frequency_hz"])
    if fps != expected_fps:
        findings.append(
            _result("FAIL", f"Dataset FPS is {fps}; expected {expected_fps}.", blocking=True)
        )
    else:
        findings.append(_result("PASS", f"Dataset FPS is {fps:g}."))
    episode_lengths: list[int] = []
    for episode_index in range(dataset.num_episodes):
        episode = episode_columns(
            dataset, episode_index, ["observation.state", "action", "timestamp", "episode_index"]
        )
        frame_count = len(episode["action"])
        episode_lengths.append(frame_count)
        errors = validate_episode_arrays(
            episode["observation.state"],
            episode["action"],
            {
                "observation.images.front": [None] * frame_count,
                "observation.images.wrist": [None] * frame_count,
            },
        )
        if errors:
            findings.append(
                _result("FAIL", f"Episode {episode_index}: {' '.join(errors)}", blocking=True)
            )
        timestamps = _as_list(episode["timestamp"])
        if timestamps and not isinstance(timestamps[0], list):
            timestamps = [[timestamp] for timestamp in timestamps]
        if len(timestamps) > 1:
            observed_step = float(timestamps[1][0]) - float(timestamps[0][0])
            if abs(observed_step - (1.0 / expected_fps)) > 1e-4:
                findings.append(
                    _result(
                        "FAIL",
                        f"Episode {episode_index}: timestamp interval {observed_step:.6f}s is not 1/{expected_fps:g}s.",
                        blocking=True,
                    )
                )
    video_paths = []
    for episode_index in range(dataset.num_episodes):
        for key in ("observation.images.front", "observation.images.wrist"):
            try:
                path = dataset.meta.get_video_file_path(episode_index, key)
            except (AttributeError, KeyError, TypeError) as exc:
                findings.append(
                    _result("FAIL", f"Cannot resolve {key} video: {exc}", blocking=True)
                )
                continue
            resolved_path = dataset.root / path
            video_paths.append(str(resolved_path))
            if not resolved_path.is_file():
                findings.append(
                    _result("FAIL", f"Missing video file: {resolved_path}", blocking=True)
                )
    if video_paths and all(Path(path).is_file() for path in video_paths):
        findings.append(_result("PASS", "Both camera video files are present for every episode."))
    return _summary(repo_id, root, findings, sidecar=sidecar, episode_lengths=episode_lengths)


def _summary(
    repo_id: str,
    root: Path,
    findings: list[dict[str, Any]],
    *,
    sidecar: dict[str, Any] | None = None,
    episode_lengths: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "repo_id": repo_id,
        "dataset_root": str(root),
        "lerobot_version": _installed_lerobot_version(),
        "valid": not any(finding["blocking"] for finding in findings),
        "findings": findings,
        "sidecar": sidecar,
        "episode_lengths": episode_lengths or [],
    }


def _installed_lerobot_version() -> str | None:
    try:
        return importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError:
        return None


def write_report(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# Recorded dataset validation", "", f"- Valid: **{summary['valid']}**", ""]
    lines.extend(
        f"- `{finding['status']}`: {finding['message']}" for finding in summary["findings"]
    )
    lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        frozen = yaml.safe_load(args.frozen_config.read_text(encoding="utf-8"))
        summary = validate_dataset(args.repo_id, args.dataset_root, frozen)
    except (OSError, ValueError, TypeError) as exc:
        summary = _summary(
            args.repo_id,
            dataset_path(args.dataset_root, args.repo_id),
            [_result("FAIL", str(exc), blocking=True)],
        )
    write_report(summary, args.output_dir)
    print(json.dumps(summary, indent=2))
    return 0 if summary["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
