"""Canonical manifests and deterministic checks for the PickOrange audit."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

BLOCKING_LEVELS = {"FAIL"}


@dataclass(frozen=True)
class Finding:
    level: str
    field: str
    expected: Any
    actual: Any
    message: str
    blocking: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "field": self.field,
            "expected": self.expected,
            "actual": self.actual,
            "message": self.message,
            "blocking": self.blocking,
        }


def get_path(data: Mapping[str, Any], path: str) -> Any:
    """Return a dotted path, or ``None`` if one of its components is absent."""

    current: Any = data
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            return None
        current = current[component]
    return current


def manifest_value(data: Mapping[str, Any], path: str) -> tuple[Any, str]:
    """Extract a manifest leaf's value and status, accepting raw runtime values."""

    item = get_path(data, path)
    if item is None:
        return None, "unknown"
    if isinstance(item, Mapping) and "value" in item:
        return item.get("value"), str(item.get("status", "unknown"))
    return item, "confirmed"


def compare_field(
    expected_manifest: Mapping[str, Any],
    actual_manifest: Mapping[str, Any],
    path: str,
    *,
    blocking: bool = True,
) -> Finding:
    """Compare one canonical field without upgrading unknown evidence to a pass."""

    expected, expected_status = manifest_value(expected_manifest, path)
    actual, actual_status = manifest_value(actual_manifest, path)
    if expected_status in {"unknown", "inferred"}:
        return Finding("UNKNOWN", path, expected, actual, "Expected value is not confirmed.")
    if actual_status in {"unknown", "inferred"} or actual is None:
        return Finding(
            "UNKNOWN",
            path,
            expected,
            actual,
            "Runtime/dataset value is unavailable or unconfirmed.",
        )
    if expected == actual:
        return Finding("PASS", path, expected, actual, "Values match.")
    return Finding(
        "FAIL" if blocking else "WARNING",
        path,
        expected,
        actual,
        "Values differ.",
        blocking=blocking,
    )


DATASET_PATHS = (
    "dataset.format_version",
    "robot.type",
    "actions.dimension",
    "actions.joint_order",
    "state.dimension",
    "state.joint_order",
    "cameras.front.dataset_key",
    "cameras.front.recorded_resolution",
    "cameras.front.codec",
    "cameras.wrist.dataset_key",
    "cameras.wrist.recorded_resolution",
    "cameras.wrist.codec",
    "simulation.dataset_frequency_hz",
)
ENVIRONMENT_PATHS = (
    "robot.type",
    "robot.asset_path",
    "robot.joint_order",
    "actions.dimension",
    "actions.joint_order",
    "actions.command_type",
    "actions.units_before_dataset_conversion",
    "cameras.front.environment_name",
    "cameras.front.prim_path",
    "cameras.front.recorded_resolution",
    "cameras.wrist.environment_name",
    "cameras.wrist.prim_path",
    "cameras.wrist.recorded_resolution",
    "simulation.control_frequency_hz",
)


def compare_manifests(
    expected: Mapping[str, Any], dataset: Mapping[str, Any], environment: Mapping[str, Any]
) -> list[Finding]:
    """Run the non-negotiable schema/action/camera/time checks."""

    findings = [compare_field(expected, dataset, path) for path in DATASET_PATHS]
    findings.extend(compare_field(expected, environment, path) for path in ENVIRONMENT_PATHS)
    return findings


def report_markdown(findings: list[Finding]) -> str:
    """Render a small, reviewable compatibility report."""

    lines = [
        "# PickOrange compatibility report",
        "",
        "| Level | Field | Result |",
        "| --- | --- | --- |",
    ]
    for finding in findings:
        result = finding.message.replace("|", "\\|")
        lines.append(f"| {finding.level} | `{finding.field}` | {result} |")
    failures = sum(finding.blocking and finding.level == "FAIL" for finding in findings)
    unknowns = sum(finding.level == "UNKNOWN" for finding in findings)
    lines.extend(["", f"Blocking failures: {failures}", f"Unknowns: {unknowns}", ""])
    return "\n".join(lines)
