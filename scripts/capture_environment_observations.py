#!/usr/bin/env python3
"""Capture raw LeIsaac cameras and the LeRobot 0.4.1 ACT input representation.

The comparison is semantic (view, framing, scale and preprocessing), never a
pixel-equality assertion. A dataset frame is optional and requires a converted
v3 dataset because LeRobot 0.4.1 cannot read v2.1 directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="LeIsaac-SO101-PickOrange-v0")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/compatibility/cameras"))
    parser.add_argument("--dataset-repo-id", help="Optional converted v3 dataset repository.")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--frozen-config", type=Path, default=Path("configs/pick_orange_current_stack.yaml")
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def to_uint8_hwc(frame: Any) -> np.ndarray:
    """Accept Isaac tensors/arrays in CHW or HWC and return an RGB image."""

    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    image = np.asarray(frame)
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in {1, 3, 4}:
        image = np.moveaxis(image, 0, -1)
    if image.dtype.kind == "f":
        image = np.clip(image * (255 if image.max(initial=0) <= 1 else 1), 0, 255).astype(np.uint8)
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image


def lerobot_act_input(frame_hwc: np.ndarray) -> np.ndarray:
    """LeRobot 0.4.1 env preprocessing: HWC uint8 -> CHW float32 in [0, 1]."""

    if frame_hwc.dtype != np.uint8 or frame_hwc.ndim != 3 or frame_hwc.shape[-1] != 3:
        raise ValueError("Expected an HWC uint8 RGB frame.")
    return np.moveaxis(frame_hwc, -1, 0).astype(np.float32) / 255.0


def _save_png(image: np.ndarray, path: Path) -> None:
    from PIL import Image

    Image.fromarray(image).save(path)


def _save_sheet(images: list[tuple[str, np.ndarray]], path: Path) -> None:
    from PIL import Image, ImageDraw

    width = max(image.shape[1] for _, image in images)
    row_height = max(image.shape[0] for _, image in images) + 24
    sheet = Image.new("RGB", (width, row_height * len(images)), "black")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(images):
        y = index * row_height
        sheet.paste(Image.fromarray(image), (0, y + 24))
        draw.text((4, y + 4), label, fill="white")
    sheet.save(path)


def main(argv: list[str] | None = None) -> int:
    try:
        from isaaclab.app import AppLauncher
    except ImportError:
        print("Camera capture requires the installed Isaac Lab and LeIsaac stack.", file=sys.stderr)
        return 3
    parser = build_parser()
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(enable_cameras=True, headless=True)
    args = parser.parse_args(argv)
    app_launcher = AppLauncher(args)
    env = None
    try:
        import gymnasium as gym
        import leisaac  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg

        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        frozen = yaml.safe_load(args.frozen_config.read_text(encoding="utf-8"))
        if frozen["environment_id"] != args.task:
            raise ValueError("--task does not match the frozen current-stack configuration.")
        env_cfg = parse_env_cfg(args.task, device="cuda", num_envs=1)
        env_cfg.use_teleop_device("so101leader")
        for name in vars(env_cfg.events):
            if name.startswith("domain_randomize_"):
                setattr(env_cfg.events, name, None)
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        records: list[dict[str, Any]] = []
        sheet_images: list[tuple[str, np.ndarray]] = []
        for seed in args.seeds:
            env.seed(seed)
            observations, _ = env.reset()
            policy_observations = observations.get("policy", observations)
            for name in ("front", "wrist"):
                raw = to_uint8_hwc(policy_observations[name])
                act_input = lerobot_act_input(raw)
                raw_path = output_dir / f"seed_{seed}_{name}_raw.png"
                act_path = output_dir / f"seed_{seed}_{name}_act_input.npy"
                _save_png(raw, raw_path)
                np.save(act_path, act_input)
                records.append(
                    {
                        "seed": seed,
                        "camera": name,
                        "raw_path": str(raw_path),
                        "raw_shape": list(raw.shape),
                        "raw_dtype": str(raw.dtype),
                        "raw_range": [int(raw.min()), int(raw.max())],
                        "act_input_path": str(act_path),
                        "act_input_shape": list(act_input.shape),
                        "act_input_dtype": str(act_input.dtype),
                        "act_input_range": [float(act_input.min()), float(act_input.max())],
                    }
                )
                sheet_images.append((f"simulation seed={seed} {name}", raw))
        if args.dataset_repo_id:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            dataset = LeRobotDataset(
                args.dataset_repo_id, root=args.dataset_root, episodes=[args.episode_index]
            )
            sample = dataset[0]
            for key in ("observation.images.front", "observation.images.wrist"):
                image = to_uint8_hwc(sample[key])
                sheet_images.append((f"dataset episode={args.episode_index} {key}", image))
        _save_sheet(sheet_images, output_dir / "comparison_sheet.png")
        (output_dir / "capture_manifest.json").write_text(
            json.dumps(records, indent=2) + "\n", encoding="utf-8"
        )
    except Exception as exc:
        print(f"Camera capture failed: {exc}", file=sys.stderr)
        return 3
    finally:
        if env is not None:
            env.close()
        app_launcher.app.close()
    print(f"Camera outputs written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
