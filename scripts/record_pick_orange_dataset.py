#!/usr/bin/env python3
"""Record keyboard/gamepad PickOrange data with the IK-produced 6D joint target.

The device command is intentionally *not* written as ``action``.  LeIsaac's
keyboard/gamepad command is eight-dimensional (relative SE(3), shoulder pan,
gripper); after ``env.step`` Isaac Lab has resolved it and exposes the six
absolute articulation targets in ``robot.data.joint_pos_target``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from so101_sorting.current_stack import environment_fingerprint

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
JOINT_FEATURE_NAMES = [f"{joint}.pos" for joint in JOINTS]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id", required=True, help="New local-only LeRobotDataset v3 repository ID."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/current_stack"))
    parser.add_argument("--teleop-device", choices=["keyboard", "gamepad"], default="keyboard")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-successes", type=int, default=1, help="Keep one for runtime validation."
    )
    parser.add_argument(
        "--frozen-config", type=Path, default=Path("configs/pick_orange_current_stack.yaml")
    )
    return parser


def _camera_frame(observations: dict[str, Any], name: str) -> Any:
    """Extract one HWC RGB frame from LeIsaac's policy observation group."""

    import numpy as np

    frame = observations.get("policy", observations)[name]
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.ndim == 3 and frame.shape[0] in {1, 3, 4} and frame.shape[-1] not in {1, 3, 4}:
        frame = np.moveaxis(frame, 0, -1)
    if frame.ndim != 3 or frame.shape[-1] not in {3, 4}:
        raise ValueError(f"Camera {name} is not an HWC/CHW RGB image: {frame.shape}")
    return frame[..., :3].astype(np.uint8, copy=False)


def _features(env: Any) -> dict[str, dict[str, Any]]:
    cameras = env.scene.sensors
    result: dict[str, dict[str, Any]] = {
        "observation.state": {"dtype": "float32", "shape": [6], "names": JOINT_FEATURE_NAMES},
        "action": {"dtype": "float32", "shape": [6], "names": JOINT_FEATURE_NAMES},
    }
    for name in ("front", "wrist"):
        height, width = cameras[name].image_shape
        result[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channels"],
        }
    return result


def _write_sidecar(
    dataset_root: Path,
    repo_id: str,
    frozen: dict[str, Any],
    seed: int,
    episodes: list[dict[str, Any]],
) -> None:
    path = dataset_root / repo_id / "meta" / "current_stack_manifest.json"
    payload = {
        "environment_fingerprint": environment_fingerprint(frozen),
        "collection_seed": seed,
        "teleop_device": "keyboard_or_gamepad_8d_relative_ik",
        "state_timing": "robot joint_pos immediately before the action transition",
        "action_timing": "robot joint_pos_target produced by Isaac Lab IK/relative action during that transition",
        "action_units": "radians, six absolute joint targets in JOINTS order",
        "sampling": "one frame every two 60 Hz control steps (30 Hz)",
        "episodes": episodes,
        "frozen_config": frozen,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Collection sidecar written to {path}")


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    try:
        from isaaclab.app import AppLauncher
    except ImportError:
        print(
            "Recording requires the installed Isaac Sim, Isaac Lab, LeIsaac and LeRobot 0.4.1 stack.",
            file=sys.stderr,
        )
        return 3
    parser = build_parser()
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(enable_cameras=True, headless=False)
    args = parser.parse_args(argv)
    frozen = yaml.safe_load(args.frozen_config.read_text(encoding="utf-8"))
    dataset_path = args.dataset_root / args.repo_id
    if dataset_path.exists():
        print(f"Refusing to overwrite existing dataset: {dataset_path}", file=sys.stderr)
        return 2
    app = AppLauncher(args).app
    env = None
    dataset = None
    try:
        import gymnasium as gym
        import leisaac  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        env_cfg = parse_env_cfg(frozen["environment_id"], device=args.device, num_envs=1)
        env_cfg.use_teleop_device(args.teleop_device)
        env_cfg.seed = args.seed
        env_cfg.recorders = None
        for name in vars(env_cfg.events):
            if name.startswith("domain_randomize_"):
                setattr(env_cfg.events, name, None)
        env = gym.make(frozen["environment_id"], cfg=env_cfg).unwrapped
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            root=dataset_path,
            fps=int(frozen["simulation"]["recording_frequency_hz"]),
            robot_type="so101_follower",
            features=_features(env),
        )
        if args.teleop_device == "keyboard":
            from leisaac.devices import SO101Keyboard

            controller = SO101Keyboard(env)
        else:
            from leisaac.devices import SO101Gamepad

            controller = SO101Gamepad(env)
        robot = env.scene["robot"]
        joint_ids = [robot.joint_names.index(name) for name in JOINTS]
        pending_success = False
        observations, _ = env.reset()
        controller.reset()
        episode_details: list[dict[str, Any]] = []

        def reject_episode() -> None:
            nonlocal observations
            dataset.clear_episode_buffer()
            observations, _ = env.reset()
            controller.reset()
            print("Episode rejected and reset.")

        def accept_episode() -> None:
            nonlocal pending_success
            pending_success = True

        controller.add_callback("R", reject_episode)
        controller.add_callback("N", accept_episode)
        controller.display_controls()
        print("B starts control. R rejects/resets. N saves a successful episode. Ctrl+C exits.")
        control_steps = 0
        while app.is_running() and len(episode_details) < args.max_successes:
            command = controller.advance()
            if pending_success:
                pending_success = False
                if dataset.episode_buffer and dataset.episode_buffer["size"]:
                    dataset.save_episode()
                    episode_details.append(
                        {
                            "episode_index": len(episode_details),
                            "success": True,
                            "frames": dataset.meta.total_frames,
                        }
                    )
                    print(f"Saved successful episode {len(episode_details)}/{args.max_successes}.")
                observations, _ = env.reset()
                controller.reset()
                control_steps = 0
                continue
            if command is None:
                env.render()
                continue
            if isinstance(command, dict):
                continue
            state_before = robot.data.joint_pos[0, joint_ids].detach().cpu().numpy().copy()
            observations_before = observations
            observations, _, _, _, _ = env.step(command)
            control_steps += 1
            if control_steps % 2:
                continue
            action_target = robot.data.joint_pos_target[0, joint_ids].detach().cpu().numpy().copy()
            dataset.add_frame(
                {
                    "observation.state": state_before,
                    "action": action_target,
                    "observation.images.front": _camera_frame(observations_before, "front"),
                    "observation.images.wrist": _camera_frame(observations_before, "wrist"),
                    "task": "Pick three oranges and place them on the plate.",
                }
            )
        if dataset.episode_buffer and dataset.episode_buffer["size"]:
            dataset.clear_episode_buffer()
        _write_sidecar(args.dataset_root, args.repo_id, frozen, args.seed, episode_details)
    except Exception as exc:
        print(f"Dataset recording failed: {exc}", file=sys.stderr)
        return 3
    finally:
        if dataset is not None:
            dataset.finalize()
        if env is not None:
            env.close()
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
