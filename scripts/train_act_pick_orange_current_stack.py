#!/usr/bin/env python3
"""Launch a guarded ACT run on a validated current-stack PickOrange dataset."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from so101_sorting.current_stack import environment_fingerprint, fingerprints_match

CONFIGS = {
    "smoke": Path("configs/act_pick_orange_current_stack_smoke.yaml"),
    "full": Path("configs/act_pick_orange_current_stack.yaml"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=CONFIGS, default="smoke")
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/current_stack"))
    parser.add_argument(
        "--dataset-validation-report",
        type=Path,
        default=Path("outputs/dataset_validation/report.json"),
    )
    parser.add_argument(
        "--frozen-config", type=Path, default=Path("configs/pick_orange_current_stack.yaml")
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--run", action="store_true", help="Start training; otherwise print the command."
    )
    return parser.parse_args(argv)


def output_dir_for(args: argparse.Namespace) -> Path:
    return (
        args.output_dir or Path("outputs/training") / f"act_pick_orange_current_stack_{args.mode}"
    )


def current_stack_training_allowed(
    report_path: Path, dataset_root: Path, repo_id: str, frozen: dict[str, Any]
) -> tuple[bool, str, str | None]:
    """Check validation verdict and fingerprint before allocating a GPU run."""

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return False, f"Dataset validation report is unavailable: {exc}", None
    if not report.get("valid"):
        return (
            False,
            "Dataset validation report is not valid; rerun validate_recorded_dataset.py.",
            None,
        )
    if report.get("repo_id") != repo_id:
        return False, "Dataset validation report belongs to another repo ID.", None
    sidecar_path = dataset_root / repo_id / "meta" / "current_stack_manifest.json"
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return False, f"Dataset collection sidecar is unavailable: {exc}", None
    expected = environment_fingerprint(frozen)
    matches, message = fingerprints_match(expected, sidecar.get("environment_fingerprint"))
    if not matches:
        return False, message, None
    return True, message, expected


def build_command(args: argparse.Namespace) -> tuple[list[str], Path]:
    output_dir = output_dir_for(args)
    if args.resume:
        checkpoint = output_dir / "checkpoints/last/pretrained_model/train_config.json"
        return ["lerobot-train", f"--config_path={checkpoint}", "--resume=true"], output_dir
    command = [
        "lerobot-train",
        f"--config_path={CONFIGS[args.mode]}",
        f"--dataset.repo_id={args.dataset_repo_id}",
        f"--dataset.root={args.dataset_root / args.dataset_repo_id}",
        f"--output_dir={output_dir}",
        f"--wandb.enable={'true' if args.wandb else 'false'}",
    ]
    return command, output_dir


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed.", file=sys.stderr)
        return False
    if not torch.cuda.is_available():
        print("CUDA is required to run this ACT baseline.", file=sys.stderr)
        return False
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    print(
        f"CUDA GPU: {torch.cuda.get_device_name(device)}; VRAM {free / 2**30:.1f}/{total / 2**30:.1f} GiB free"
    )
    return True


def _runtime_allowed(args: argparse.Namespace, output_dir: Path) -> tuple[bool, str | None]:
    frozen = yaml.safe_load(args.frozen_config.read_text(encoding="utf-8"))
    allowed, message, fingerprint = current_stack_training_allowed(
        args.dataset_validation_report, args.dataset_root, args.dataset_repo_id, frozen
    )
    if not allowed:
        print(f"Refusing training: {message}", file=sys.stderr)
        return False, None
    if shutil.which("lerobot-train") is None:
        print(
            "lerobot-train is not installed; install the pinned LeRobot 0.4.1 runtime.",
            file=sys.stderr,
        )
        return False, None
    try:
        version = importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError:
        version = None
    if version != "0.4.1":
        print(f"LeRobot 0.4.1 is required; found {version or 'not installed'}.", file=sys.stderr)
        return False, None
    if not _cuda_available():
        return False, None
    if args.resume:
        if not (output_dir / "checkpoints/last/pretrained_model/train_config.json").is_file():
            print("Cannot resume: last checkpoint configuration is missing.", file=sys.stderr)
            return False, None
    elif output_dir.exists():
        print(f"Refusing to overwrite existing training directory: {output_dir}", file=sys.stderr)
        return False, None
    return True, fingerprint


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(list(argv) if argv is not None else None)
    command, output_dir = build_command(args)
    print("Prepared command:\n  " + " ".join(command))
    if not args.run:
        print("Dry run only. The validation/fingerprint gate runs when --run is supplied.")
        return 0
    allowed, fingerprint = _runtime_allowed(args, output_dir)
    if not allowed:
        return 3
    log_dir = Path("outputs/training_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    (log_dir / f"act_current_stack_{stamp}.json").write_text(
        json.dumps(
            {
                "environment_fingerprint": fingerprint,
                "dataset_repo_id": args.dataset_repo_id,
                "requested_config": str(CONFIGS[args.mode]),
                "command": command,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with (log_dir / f"act_current_stack_{stamp}.log").open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
