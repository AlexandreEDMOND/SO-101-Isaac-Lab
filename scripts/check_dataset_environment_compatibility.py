#!/usr/bin/env python3
"""Compare dataset metadata, expected historical contract, and runtime environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from so101_sorting.compatibility import compare_manifests, report_markdown


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected", type=Path, default=Path("configs/pick_orange_compatibility.yaml")
    )
    parser.add_argument(
        "--dataset-manifest", type=Path, default=Path("outputs/compatibility/dataset_manifest.json")
    )
    parser.add_argument(
        "--environment-manifest",
        type=Path,
        default=Path("outputs/compatibility/environment_manifest.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/compatibility"))
    return parser.parse_args(argv)


def run_check(expected: dict, dataset: dict, environment: dict) -> tuple[dict, int]:
    findings = compare_manifests(expected, dataset, environment)
    serialized = [finding.as_dict() for finding in findings]
    blocking_failures = [
        item for item in serialized if item["level"] == "FAIL" and item["blocking"]
    ]
    result = {
        "verdict": "FAIL" if blocking_failures else "REVIEW_REQUIRED",
        "blocking_failures": len(blocking_failures),
        "unknowns": sum(item["level"] == "UNKNOWN" for item in serialized),
        "findings": serialized,
    }
    return result, 2 if blocking_failures else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    required = (args.expected, args.dataset_manifest)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print("Compatibility check requires: " + ", ".join(missing))
        return 3
    expected = yaml.safe_load(args.expected.read_text(encoding="utf-8"))
    dataset = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    environment_available = args.environment_manifest.is_file()
    environment = (
        json.loads(args.environment_manifest.read_text(encoding="utf-8"))
        if environment_available
        else {}
    )
    result, return_code = run_check(expected, dataset, environment)
    if not environment_available:
        result["verdict"] = "UNKNOWN"
        result["environment_manifest_missing"] = str(args.environment_manifest)
        return_code = 3
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "compatibility_report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    from so101_sorting.compatibility import Finding

    findings = [Finding(**item) for item in result["findings"]]
    (args.output_dir / "compatibility_report.md").write_text(
        report_markdown(findings), encoding="utf-8"
    )
    print(f"Compatibility verdict: {result['verdict']}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
