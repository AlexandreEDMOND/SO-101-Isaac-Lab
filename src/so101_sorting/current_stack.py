"""Small reproducibility primitives for the current-stack PickOrange baseline."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any


def canonical_json(data: Mapping[str, Any]) -> str:
    """Serialize configuration deterministically for a cross-machine fingerprint."""

    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def environment_fingerprint(config: Mapping[str, Any]) -> str:
    """Fingerprint only the environment contract, never paths or generated timestamps."""

    excluded = {"environment_fingerprint", "generated_at", "output_dir"}
    payload = {key: value for key, value in config.items() if key not in excluded}
    return f"sha256:{hashlib.sha256(canonical_json(payload).encode()).hexdigest()}"


def fingerprints_match(expected: str | None, actual: str | None) -> tuple[bool, str]:
    """Compare required fingerprints and return an actionable diagnostic."""

    if not expected:
        return False, "Missing expected environment fingerprint."
    if not actual:
        return False, "Missing actual environment fingerprint."
    if expected != actual:
        return False, f"Fingerprint mismatch: expected {expected}, got {actual}."
    return True, "Environment fingerprints match."


def validate_episode_arrays(
    state: Any, action: Any, camera_frames: Mapping[str, Any], *, expected_frames: int | None = None
) -> list[str]:
    """Validate array-like episode payloads without depending on Isaac or LeRobot."""

    errors: list[str] = []
    state_rows = _rows(state)
    action_rows = _rows(action)
    state_shape = _shape(state_rows)
    action_shape = _shape(action_rows)
    if len(state_shape) != 2 or state_shape[1] != 6:
        errors.append(f"observation.state must have shape [T, 6], got {state_shape}.")
    if len(action_shape) != 2 or action_shape[1] != 6:
        errors.append(f"action must have shape [T, 6], got {action_shape}.")
    if len(state_rows) != len(action_rows):
        errors.append("State and action frame counts differ.")
    if not all(math.isfinite(float(value)) for row in state_rows + action_rows for value in row):
        errors.append("State or action contains NaN/Inf.")
    frames = expected_frames if expected_frames is not None else len(state_rows)
    for key in ("observation.images.front", "observation.images.wrist"):
        values = camera_frames.get(key)
        if values is None:
            errors.append(f"Missing camera: {key}.")
            continue
        if len(values) != frames:
            errors.append(f"Camera {key} has {len(values)} frames; expected {frames}.")
    return errors


def _rows(values: Any) -> list[list[Any]]:
    """Convert Torch/NumPy/list matrix-like values without a hard NumPy dependency."""

    if hasattr(values, "detach"):
        values = values.detach().cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    return values if isinstance(values, list) else []


def _shape(values: list[Any]) -> tuple[int, ...]:
    if not isinstance(values, list):
        return ()
    if not values:
        return (0,)
    if not isinstance(values[0], list):
        return (len(values),)
    return (len(values), len(values[0]))
