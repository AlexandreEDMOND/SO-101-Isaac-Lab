from __future__ import annotations

from pathlib import Path

import yaml

from scripts.check_dataset_environment_compatibility import run_check
from so101_sorting.compatibility import compare_field, compare_manifests, report_markdown


def _node(value: object) -> dict[str, object]:
    return {"value": value, "status": "confirmed", "source": "test", "notes": ""}


def _manifest() -> dict[str, object]:
    return {
        "dataset": {"format_version": _node("v2.1")},
        "robot": {
            "type": _node("so101_follower"),
            "asset_path": _node("robot.usd"),
            "joint_order": _node(["a", "b"]),
        },
        "actions": {
            "dimension": _node(2),
            "joint_order": _node(["a.pos", "b.pos"]),
            "command_type": _node("absolute_joint_position"),
            "units_before_dataset_conversion": _node("radians"),
        },
        "state": {"dimension": _node(2), "joint_order": _node(["a.pos", "b.pos"])},
        "cameras": {
            "front": {
                "dataset_key": _node("observation.images.front"),
                "environment_name": _node("front"),
                "prim_path": _node("/front"),
                "recorded_resolution": _node([480, 640, 3]),
                "codec": _node("av1/yuv420p"),
            },
            "wrist": {
                "dataset_key": _node("observation.images.wrist"),
                "environment_name": _node("wrist"),
                "prim_path": _node("/wrist"),
                "recorded_resolution": _node([480, 640, 3]),
                "codec": _node("av1/yuv420p"),
            },
        },
        "simulation": {"dataset_frequency_hz": _node(30), "control_frequency_hz": _node(30)},
    }


def test_joint_order_mismatch_is_blocking_fail() -> None:
    expected = _manifest()
    actual = _manifest()
    actual["actions"]["joint_order"] = _node(["b.pos", "a.pos"])
    finding = compare_field(expected, actual, "actions.joint_order")
    assert finding.level == "FAIL"
    assert finding.blocking


def test_radians_degrees_mismatch_is_detected() -> None:
    expected = _manifest()
    actual = _manifest()
    actual["actions"]["units_before_dataset_conversion"] = _node("degrees")
    assert (
        compare_field(expected, actual, "actions.units_before_dataset_conversion").level == "FAIL"
    )


def test_absolute_delta_mismatch_is_detected() -> None:
    expected = _manifest()
    actual = _manifest()
    actual["actions"]["command_type"] = _node("delta_joint_position")
    assert compare_field(expected, actual, "actions.command_type").level == "FAIL"


def test_camera_and_frequency_mismatches_are_detected() -> None:
    expected = _manifest()
    actual = _manifest()
    actual["cameras"]["front"]["recorded_resolution"] = _node([240, 320, 3])
    actual["simulation"]["control_frequency_hz"] = _node(60)
    assert compare_field(expected, actual, "cameras.front.recorded_resolution").level == "FAIL"
    assert compare_field(expected, actual, "simulation.control_frequency_hz").level == "FAIL"


def test_markdown_and_return_code_for_blocking_failure() -> None:
    expected = _manifest()
    dataset = _manifest()
    environment = _manifest()
    environment["actions"]["joint_order"] = _node(["b.pos", "a.pos"])
    result, code = run_check(expected, dataset, environment)
    findings = compare_manifests(expected, dataset, environment)
    markdown = report_markdown(findings)
    assert code == 2
    assert result["blocking_failures"] >= 1
    assert "PickOrange compatibility report" in markdown


def test_compatibility_yaml_camera_extrinsics_and_intrinsics_parse() -> None:
    manifest = yaml.safe_load(
        Path("configs/pick_orange_compatibility.yaml").read_text(encoding="utf-8")
    )
    front = manifest["cameras"]["front"]
    assert len(front["extrinsics_ros_wxyz"]["value"]["position"]) == 3
    assert len(front["extrinsics_ros_wxyz"]["value"]["quaternion_wxyz"]) == 4
    assert front["intrinsics"]["value"]["focal_length_mm"] > 0
