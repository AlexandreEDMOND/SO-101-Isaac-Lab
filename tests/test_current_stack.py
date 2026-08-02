from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.evaluate_act_pick_orange import evaluation_preflight
from scripts.train_act_pick_orange_current_stack import current_stack_training_allowed
from so101_sorting.current_stack import (
    environment_fingerprint,
    fingerprints_match,
    validate_episode_arrays,
)


def _frozen() -> dict[str, object]:
    return yaml.safe_load(
        Path("configs/pick_orange_current_stack.yaml").read_text(encoding="utf-8")
    )


def test_current_stack_fingerprint_is_stable_and_in_config() -> None:
    frozen = _frozen()
    fingerprint = environment_fingerprint(frozen)
    assert fingerprint == frozen["environment_fingerprint"]
    assert fingerprint == environment_fingerprint(dict(reversed(list(frozen.items()))))


def test_fingerprint_detects_camera_and_frequency_change() -> None:
    frozen = _frozen()
    changed_camera = yaml.safe_load(yaml.safe_dump(frozen))
    changed_camera["cameras"]["front"]["resolution_hwc"] = [240, 320, 3]
    changed_frequency = yaml.safe_load(yaml.safe_dump(frozen))
    changed_frequency["simulation"]["recording_frequency_hz"] = 20
    assert environment_fingerprint(changed_camera) != environment_fingerprint(frozen)
    assert environment_fingerprint(changed_frequency) != environment_fingerprint(frozen)


def test_fingerprint_detects_joint_order_change() -> None:
    frozen = _frozen()
    changed = yaml.safe_load(yaml.safe_dump(frozen))
    changed["actions"]["order"] = list(reversed(changed["actions"]["order"]))
    assert environment_fingerprint(changed) != environment_fingerprint(frozen)


def test_episode_validation_detects_joint_order_shape_and_camera_error() -> None:
    errors = validate_episode_arrays(
        [[0.0] * 5], [[0.0] * 6], {"observation.images.front": [None]}, expected_frames=1
    )
    assert "observation.state must have shape [T, 6]" in errors[0]
    assert any("Missing camera" in error for error in errors)


def test_fingerprint_mismatch_refuses_evaluation() -> None:
    frozen = _frozen()
    allowed, findings, _ = evaluation_preflight(
        frozen,
        {"environment_fingerprint": "sha256:environment"},
        {"environment_fingerprint": environment_fingerprint(frozen)},
        {"environment_fingerprint": environment_fingerprint(frozen)},
    )
    assert not allowed
    assert findings and findings[0].startswith("environment:")
    assert not fingerprints_match("sha256:a", "sha256:b")[0]


def test_invalid_dataset_validation_refuses_training(tmp_path: Path) -> None:
    frozen = _frozen()
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"valid": False, "repo_id": "org/data"}), encoding="utf-8")
    allowed, message, _ = current_stack_training_allowed(report, tmp_path, "org/data", frozen)
    assert not allowed
    assert "not valid" in message


def test_valid_dataset_validation_and_sidecar_allow_training(tmp_path: Path) -> None:
    frozen = _frozen()
    fingerprint = environment_fingerprint(frozen)
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"valid": True, "repo_id": "org/data"}), encoding="utf-8")
    sidecar = tmp_path / "org/data/meta/current_stack_manifest.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(json.dumps({"environment_fingerprint": fingerprint}), encoding="utf-8")
    allowed, _, actual = current_stack_training_allowed(report, tmp_path, "org/data", frozen)
    assert allowed
    assert actual == fingerprint
