#!/usr/bin/env python3
"""Launch (only with --run) a reproducible LeRobot 0.4.1 ACT baseline."""

from __future__ import annotations

import argparse
import importlib.metadata
import shutil
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

CONFIGS = {
    "smoke": Path("configs/act_pick_orange_smoke.yaml"),
    "full": Path("configs/act_pick_orange.yaml"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=CONFIGS, default="smoke")
    parser.add_argument(
        "--dataset-repo-id", required=True, help="Converted LeRobotDataset v3 repository ID."
    )
    parser.add_argument("--dataset-root", type=Path, help="Optional local v3 dataset root.")
    parser.add_argument(
        "--require-compatibility-report",
        action="store_true",
        help="Refuse --run when the report contains a blocking FAIL.",
    )
    parser.add_argument(
        "--compatibility-report",
        type=Path,
        default=Path("outputs/compatibility/compatibility_report.json"),
        help="JSON output from check_dataset_environment_compatibility.py.",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Override output_dir from the YAML configuration."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from output-dir/checkpoints/last."
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Enable optional Weights & Biases logging."
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually start lerobot-train (otherwise print the command).",
    )
    return parser.parse_args(argv)


def _cuda_report() -> bool:
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed.", file=sys.stderr)
        return False
    if not torch.cuda.is_available():
        print(
            "CUDA is unavailable. This ACT baseline is configured for an NVIDIA GPU.",
            file=sys.stderr,
        )
        return False
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    print(
        f"CUDA GPU: {torch.cuda.get_device_name(device)} ({properties.total_memory / 2**30:.1f} GiB total)"
    )
    print(f"CUDA VRAM available: {free_bytes / 2**30:.1f} / {total_bytes / 2**30:.1f} GiB")
    return True


def _default_output_dir(mode: str) -> Path:
    return Path("outputs/training") / f"act_pick_orange_{mode}"


def latest_checkpoint_config(output_dir: Path) -> Path | None:
    """Return the newest local LeRobot checkpoint configuration, if any."""

    checkpoint_root = output_dir / "checkpoints"
    legacy = checkpoint_root / "last/pretrained_model/train_config.json"
    if legacy.is_file():
        return legacy
    candidates = list(checkpoint_root.glob("*/pretrained_model/train_config.json"))
    if not candidates:
        return None

    def checkpoint_order(path: Path) -> int:
        try:
            return int(path.parents[1].name)
        except ValueError:
            return -1

    return max(candidates, key=checkpoint_order)


def compatibility_report_allows_training(report_path: Path) -> tuple[bool, str]:
    """Return whether a report has no explicitly blocking compatibility failure."""

    if not report_path.is_file():
        return False, f"Compatibility report not found: {report_path}"
    try:
        import json

        report = json.loads(report_path.read_text(encoding="utf-8"))
        failures = int(report.get("blocking_failures", 0))
    except (OSError, ValueError, TypeError) as exc:
        return False, f"Cannot read compatibility report: {exc}"
    if failures:
        return (
            False,
            f"Refusing training: compatibility report contains {failures} blocking FAIL finding(s).",
        )
    return True, "Compatibility report has no blocking FAIL finding."


def build_command(args: argparse.Namespace) -> tuple[list[str], Path]:
    output_dir = args.output_dir or _default_output_dir(args.mode)
    if args.resume:
        checkpoint_config = latest_checkpoint_config(output_dir)
        if checkpoint_config is None:
            checkpoint_config = output_dir / "checkpoints/last/pretrained_model/train_config.json"
        command = ["lerobot-train", f"--config_path={checkpoint_config}", "--resume=true"]
    else:
        command = [
            "lerobot-train",
            f"--config_path={CONFIGS[args.mode]}",
            f"--dataset.repo_id={args.dataset_repo_id}",
            f"--output_dir={output_dir}",
        ]
        if args.dataset_root is not None:
            command.append(f"--dataset.root={args.dataset_root}")
    command.append(f"--wandb.enable={'true' if args.wandb else 'false'}")
    return command, output_dir


def _validate_runtime(args: argparse.Namespace, output_dir: Path) -> bool:
    if args.require_compatibility_report:
        allowed, message = compatibility_report_allows_training(args.compatibility_report)
        if not allowed:
            print(message, file=sys.stderr)
            return False
    if shutil.which("lerobot-train") is None:
        print(
            "lerobot-train was not found. Install LeRobot 0.4.1 in the active Python environment.",
            file=sys.stderr,
        )
        return False
    try:
        version = importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError:
        print("LeRobot is not installed in the active Python environment.", file=sys.stderr)
        return False
    if version != "0.4.1":
        print(f"LeRobot {version} detected; this baseline is pinned to 0.4.1.", file=sys.stderr)
        return False
    if not _cuda_report():
        return False
    if args.resume:
        checkpoint_config = latest_checkpoint_config(output_dir)
        if checkpoint_config is None:
            print(
                f"Cannot resume: checkpoint configuration not found under {output_dir / 'checkpoints'}.",
                file=sys.stderr,
            )
            return False
    elif output_dir.exists():
        print(
            f"Refusing to overwrite existing training directory: {output_dir}. Use --resume or a new --output-dir.",
            file=sys.stderr,
        )
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(list(argv) if argv is not None else None)
    command, output_dir = build_command(args)
    print("Prepared command:\n  " + " ".join(command))
    if not args.run:
        print("Dry run only. Add --run to start training.")
        return 0
    if not _validate_runtime(args, output_dir):
        return 3

    log_dir = Path("outputs/training_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"act_pick_orange_{args.mode}_{timestamp}.log"
    requested_config = log_dir / f"act_pick_orange_{args.mode}_{timestamp}.yaml"
    if not args.resume:
        requested_config.write_text(
            CONFIGS[args.mode].read_text(encoding="utf-8"), encoding="utf-8"
        )
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"# started_at={datetime.now(UTC).isoformat()}\n")
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
        log_file.write(f"# finished_at={datetime.now(UTC).isoformat()}\n")
    print(f"LeRobot output saved to {log_path}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
