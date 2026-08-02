#!/usr/bin/env python3
"""Create local loss plots and a concise report from a LeRobot training run."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from so101_sorting.baseline import parse_lerobot_training_metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-dir",
        type=Path,
        required=True,
        help="LeRobot output_dir containing checkpoints.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        required=True,
        help="Captured stdout from train_act_pick_orange.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/training_analysis"),
        help="Analysis directory.",
    )
    return parser.parse_args(argv)


def _checkpoint_summary(training_dir: Path) -> dict[str, Any]:
    checkpoints = training_dir / "checkpoints"
    if not checkpoints.exists():
        return {
            "available": [],
            "selected": None,
            "selection_reason": "No checkpoints directory found.",
        }
    available = sorted(path.name for path in checkpoints.iterdir() if path.is_dir())
    last = checkpoints / "last"
    if last.exists():
        return {
            "available": available,
            "selected": str(last),
            "selection_reason": "Latest checkpoint. No validation loss was recorded, so a best checkpoint cannot be inferred.",
        }
    numeric = sorted((name for name in available if name.isdigit()), key=int)
    selected = checkpoints / numeric[-1] if numeric else None
    return {
        "available": available,
        "selected": str(selected) if selected else None,
        "selection_reason": "Highest numbered checkpoint; no validation loss was recorded.",
    }


def _find_config(training_dir: Path) -> str | None:
    candidate = training_dir / "checkpoints/last/pretrained_model/train_config.json"
    return str(candidate) if candidate.is_file() else None


def _duration_from_log(lines: list[str]) -> float | None:
    timestamps: dict[str, datetime] = {}
    for line in lines:
        if line.startswith("# started_at=") or line.startswith("# finished_at="):
            key, value = line[2:].strip().split("=", maxsplit=1)
            try:
                timestamps[key] = datetime.fromisoformat(value)
            except ValueError:
                continue
    if "started_at" in timestamps and "finished_at" in timestamps:
        return max(0.0, (timestamps["finished_at"] - timestamps["started_at"]).total_seconds())
    return None


def _svg_loss_curve(metrics: list[dict[str, float | int]], path: Path) -> None:
    width, height, padding = 800, 360, 45
    if not metrics:
        path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="360"><text x="20" y="40">No loss values found.</text></svg>\n',
            encoding="utf-8",
        )
        return
    steps = [float(metric["step"]) for metric in metrics]
    losses = [float(metric["loss"]) for metric in metrics]
    min_step, max_step = min(steps), max(steps)
    min_loss, max_loss = min(losses), max(losses)
    step_span = max(max_step - min_step, 1.0)
    loss_span = max(max_loss - min_loss, 1e-12)
    points = " ".join(
        f"{padding + (step - min_step) / step_span * (width - 2 * padding):.2f},"
        f"{height - padding - (loss - min_loss) / loss_span * (height - 2 * padding):.2f}"
        for step, loss in zip(steps, losses, strict=True)
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/><path d="M {padding} {padding} V {height - padding} H {width - padding}" stroke="black" fill="none"/>
<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="2"/><text x="{padding}" y="22">Training loss</text>
<text x="{padding}" y="{height - 12}">step {int(min_step)}–{int(max_step)}</text><text x="{width - 180}" y="22">loss {min_loss:.5g}–{max_loss:.5g}</text></svg>\n'''
    path.write_text(svg, encoding="utf-8")


def analyze(training_dir: Path, log_file: Path, output_dir: Path) -> dict[str, Any]:
    if not log_file.is_file():
        raise FileNotFoundError(f"Training log not found: {log_file}")
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    metrics = parse_lerobot_training_metrics(lines)
    _svg_loss_curve(metrics, output_dir / "training_loss.svg")
    duration_seconds = _duration_from_log(lines)
    summary: dict[str, Any] = {
        "training_dir": str(training_dir),
        "log_file": str(log_file),
        "analyzed_at": datetime.now(UTC).isoformat(),
        "duration_seconds": duration_seconds,
        "total_logged_steps": int(metrics[-1]["step"]) if metrics else 0,
        "loss": {
            "points": len(metrics),
            "first": float(metrics[0]["loss"]) if metrics else None,
            "last": float(metrics[-1]["loss"]) if metrics else None,
            "minimum": min((float(item["loss"]) for item in metrics), default=None),
        },
        "validation_loss": None,
        "checkpoint": _checkpoint_summary(training_dir),
        "effective_config": _find_config(training_dir),
        "artifacts": {"training_loss_svg": str(output_dir / "training_loss.svg")},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    markdown = "\n".join(
        [
            "# Training analysis",
            "",
            f"- Logged steps: {summary['total_logged_steps']}",
            f"- Last training loss: {summary['loss']['last']}",
            f"- Minimum training loss: {summary['loss']['minimum']}",
            f"- Validation loss: {summary['validation_loss'] or 'not logged'}",
            f"- Selected checkpoint: {summary['checkpoint']['selected'] or 'none'}",
            f"- Reason: {summary['checkpoint']['selection_reason']}",
            f"- Effective configuration: {summary['effective_config'] or 'not found'}",
            "",
        ]
    )
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = analyze(args.training_dir, args.log_file, args.output_dir)
    except (FileNotFoundError, OSError) as exc:
        print(f"Analysis failed: {exc}")
        return 2
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
