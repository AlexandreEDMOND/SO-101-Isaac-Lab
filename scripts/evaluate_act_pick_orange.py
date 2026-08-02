#!/usr/bin/env python3
"""Preflight a fingerprint-safe ACT evaluation in current-stack PickOrange.

LeIsaac v0.4.0's published evaluation client targets the v0.3.3 remote-policy
protocol.  It is not an official direct adapter for LeRobot 0.4.1 ACT.  This
entry point therefore performs all reproducibility checks and refuses rollout
until that adapter has been validated on the installed stack.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from so101_sorting.current_stack import environment_fingerprint, fingerprints_match


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/current_stack"))
    parser.add_argument(
        "--environment-manifest",
        type=Path,
        default=Path("outputs/compatibility/environment_manifest.json"),
    )
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument(
        "--frozen-config", type=Path, default=Path("configs/pick_orange_current_stack.yaml")
    )
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--allow-fingerprint-mismatch",
        action="store_true",
        help="Diagnostic only; never use for a metric.",
    )
    parser.add_argument(
        "--run", action="store_true", help="Reserved for a validated LeRobot 0.4.1 adapter."
    )
    return parser.parse_args(argv)


def evaluation_preflight(
    frozen: dict[str, Any],
    environment_manifest: dict[str, Any],
    dataset_sidecar: dict[str, Any],
    training_manifest: dict[str, Any],
) -> tuple[bool, list[str], str]:
    fingerprint = environment_fingerprint(frozen)
    messages: list[str] = []
    for name, actual in {
        "environment": environment_manifest.get("environment_fingerprint"),
        "dataset": dataset_sidecar.get("environment_fingerprint"),
        "training": training_manifest.get("environment_fingerprint"),
    }.items():
        matches, message = fingerprints_match(fingerprint, actual)
        if not matches:
            messages.append(f"{name}: {message}")
    return not messages, messages, fingerprint


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        frozen = yaml.safe_load(args.frozen_config.read_text(encoding="utf-8"))
        environment = json.loads(args.environment_manifest.read_text(encoding="utf-8"))
        sidecar = json.loads(
            (
                args.dataset_root / args.dataset_repo_id / "meta" / "current_stack_manifest.json"
            ).read_text(encoding="utf-8")
        )
        training = json.loads(args.training_manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"Evaluation preflight failed: {exc}")
        return 2
    allowed, messages, fingerprint = evaluation_preflight(frozen, environment, sidecar, training)
    report = {
        "allowed": allowed,
        "environment_fingerprint": fingerprint,
        "checkpoint": str(args.checkpoint),
        "num_episodes": args.num_episodes,
        "seed_start": args.seed_start,
        "headless": args.headless,
        "findings": messages,
    }
    print(json.dumps(report, indent=2))
    if not allowed and not args.allow_fingerprint_mismatch:
        return 3
    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        return 2
    if args.run:
        print(
            "Refusing rollout: no official, validated LeIsaac 0.4.0 -> LeRobot 0.4.1 ACT adapter is available. "
            "Validate an adapter against the recorded dataset before enabling metrics."
        )
        return 3
    print(
        "Preflight only. Use --run only after the adapter documented in pick_orange_compatibility.md is validated."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
