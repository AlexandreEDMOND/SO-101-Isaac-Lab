"""Small, dependency-light helpers for the PickOrange baseline scripts.

The helpers deliberately operate on metadata and text logs only.  Dataset and
simulation access stays in the command-line scripts, where optional heavy
dependencies can produce actionable error messages.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

DATASET_SOURCE_REPO_ID = "LightwheelAI/leisaac-pick-orange"
REQUIRED_CAMERA_KEYS = ("observation.images.front", "observation.images.wrist")
REQUIRED_VECTOR_KEYS = ("observation.state", "action")
EXPECTED_JOINT_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


@dataclass(frozen=True)
class DatasetValidation:
    """Result of schema checks that can be run before loading frame data."""

    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def is_compatible_schema(self) -> bool:
        return not self.errors


def validate_pick_orange_metadata(info: Mapping[str, Any]) -> DatasetValidation:
    """Check the fields required by the SO-101 ACT baseline.

    This validates the published metadata, not numerical samples.  Numerical
    validation is performed by ``scripts/inspect_dataset.py`` after a v3
    dataset can be opened through the official LeRobot API.
    """

    errors: list[str] = []
    warnings: list[str] = []
    features = info.get("features")
    if not isinstance(features, Mapping):
        return DatasetValidation(("meta/info.json does not contain a features mapping.",), ())

    for key in REQUIRED_VECTOR_KEYS:
        feature = features.get(key)
        if not isinstance(feature, Mapping):
            errors.append(f"Missing required feature: {key}.")
            continue
        if feature.get("dtype") != "float32":
            errors.append(f"{key} must use float32, got {feature.get('dtype')!r}.")
        if list(feature.get("shape", [])) != [6]:
            errors.append(f"{key} must have shape [6], got {feature.get('shape')!r}.")
        names = tuple(feature.get("names", []))
        if names != EXPECTED_JOINT_NAMES:
            errors.append(f"{key} joint names differ from the expected SO-101 order: {names!r}.")

    for key in REQUIRED_CAMERA_KEYS:
        feature = features.get(key)
        if not isinstance(feature, Mapping):
            errors.append(f"Missing required camera feature: {key}.")
            continue
        if feature.get("dtype") not in {"video", "image"}:
            errors.append(f"{key} must be an image or video feature, got {feature.get('dtype')!r}.")

    if "task_index" not in features:
        errors.append("Missing task_index feature.")
    if not info.get("fps"):
        errors.append("Dataset metadata does not define a positive fps value.")
    if info.get("codebase_version") != "v3.0":
        warnings.append(
            "This dataset is not in LeRobotDataset v3.0 format; LeRobot 0.4.1 requires conversion."
        )
    return DatasetValidation(tuple(errors), tuple(warnings))


def summarize_episode_lengths(lengths: Sequence[int], fps: float) -> dict[str, float | int]:
    """Return frame and second duration statistics for non-empty episode lengths."""

    valid_lengths = [length for length in lengths if length >= 0]
    if not valid_lengths:
        raise ValueError("No episode lengths were provided.")
    if fps <= 0:
        raise ValueError("fps must be positive.")
    return {
        "episodes": len(valid_lengths),
        "min_frames": min(valid_lengths),
        "mean_frames": sum(valid_lengths) / len(valid_lengths),
        "max_frames": max(valid_lengths),
        "min_seconds": min(valid_lengths) / fps,
        "mean_seconds": sum(valid_lengths) / len(valid_lengths) / fps,
        "max_seconds": max(valid_lengths) / fps,
    }


_METRIC_PATTERN = re.compile(
    r"(?:step:\s*(?P<step>\d+)).*?(?:loss:\s*(?P<loss>[-+\deE.]+))"
    r"(?:.*?(?:lr:\s*(?P<lr>[-+\deE.]+)))?"
)


def parse_lerobot_training_metrics(lines: Iterable[str]) -> list[dict[str, float | int]]:
    """Parse the stable textual metrics emitted by LeRobot 0.4.1's trainer."""

    metrics: list[dict[str, float | int]] = []
    for line in lines:
        match = _METRIC_PATTERN.search(line)
        if not match:
            continue
        loss = float(match.group("loss"))
        if not math.isfinite(loss):
            continue
        entry: dict[str, float | int] = {"step": int(match.group("step")), "loss": loss}
        if match.group("lr") is not None:
            entry["learning_rate"] = float(match.group("lr"))
        metrics.append(entry)
    return metrics


def validate_act_config(config: Mapping[str, Any]) -> list[str]:
    """Validate the subset of LeRobot 0.4.1 config fields we intentionally use."""

    errors: list[str] = []
    dataset = config.get("dataset")
    policy = config.get("policy")
    if not isinstance(dataset, Mapping) or not dataset.get("repo_id"):
        errors.append("dataset.repo_id is required.")
    if not isinstance(policy, Mapping) or policy.get("type") != "act":
        errors.append("policy.type must be 'act'.")
    if isinstance(policy, Mapping):
        chunk_size = policy.get("chunk_size")
        action_steps = policy.get("n_action_steps")
        if not isinstance(chunk_size, int) or chunk_size < 1:
            errors.append("policy.chunk_size must be a positive integer.")
        if not isinstance(action_steps, int) or action_steps < 1:
            errors.append("policy.n_action_steps must be a positive integer.")
        elif isinstance(chunk_size, int) and action_steps > chunk_size:
            errors.append("policy.n_action_steps cannot exceed policy.chunk_size.")
    for key in ("batch_size", "steps", "save_freq", "log_freq", "seed", "num_workers"):
        if not isinstance(config.get(key), int):
            errors.append(f"{key} must be an integer.")
    return errors
