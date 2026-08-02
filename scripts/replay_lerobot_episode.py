#!/usr/bin/env python3
"""Run an explicitly non-deterministic action replay of a converted PickOrange episode.

The published dataset has no scene state, object poses, seed or historical
simulator pin. This tool therefore never calls its result an exact episode
replay: it only tests the action-unit/order/time adapter against a fresh reset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--action-rate-hz", type=float, required=True)
    parser.add_argument(
        "--seed", type=int, default=42, help="Fresh-reset seed; not a restored dataset seed."
    )
    parser.add_argument("--task", default="LeIsaac-SO101-PickOrange-v0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/compatibility/replay"))
    parser.add_argument(
        "--allow-nondeterministic-action-replay",
        action="store_true",
        help="Required acknowledgement that object initial state cannot be restored from this dataset.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def trajectory_metrics(expected: np.ndarray, actual: np.ndarray) -> dict[str, object]:
    if expected.shape != actual.shape:
        raise ValueError(f"Trajectory shape mismatch: {expected.shape} != {actual.shape}")
    error = actual - expected
    return {
        "joint_rmse": np.sqrt(np.mean(np.square(error), axis=0)).tolist(),
        "joint_max_abs_error": np.max(np.abs(error), axis=0).tolist(),
        "global_rmse": float(np.sqrt(np.mean(np.square(error)))),
    }


def main(argv: list[str] | None = None) -> int:
    try:
        from isaaclab.app import AppLauncher
    except ImportError:
        print("Action replay requires the installed Isaac Lab and LeIsaac stack.", file=sys.stderr)
        return 3
    parser = build_parser()
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(enable_cameras=True, headless=True)
    args = parser.parse_args(argv)
    if not args.allow_nondeterministic_action_replay:
        print(
            "Refusing replay: add --allow-nondeterministic-action-replay after reading the limitation."
        )
        return 2
    app_launcher = AppLauncher(args)
    env = None
    try:
        import gymnasium as gym
        import leisaac  # noqa: F401
        import torch
        from isaaclab_tasks.utils import parse_env_cfg
        from leisaac.utils.robot_utils import convert_lerobot_action_to_leisaac
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(
            args.dataset_repo_id, root=args.dataset_root, episodes=[args.episode_index]
        )
        if dataset.fps != args.action_rate_hz:
            print(
                f"Dataset metadata is {dataset.fps} Hz but replay was requested at {args.action_rate_hz} Hz. "
                "No implicit resampling is performed.",
                file=sys.stderr,
            )
            return 2
        actions = np.stack(
            [dataset[index]["action"].cpu().numpy() for index in range(len(dataset))]
        )
        expected_state = np.stack(
            [dataset[index]["observation.state"].cpu().numpy() for index in range(len(dataset))]
        )
        env_cfg = parse_env_cfg(args.task, device="cuda", num_envs=1)
        env_cfg.use_teleop_device("so101leader")
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        env.seed(args.seed)
        env.reset()
        replay_states: list[np.ndarray] = []
        for action in actions:
            radians = convert_lerobot_action_to_leisaac(action[None, :])
            env_action = torch.as_tensor(radians, dtype=torch.float32, device=env.device)
            env.step(env_action)
            replay_states.append(env.scene["robot"].data.joint_pos[0, :6].detach().cpu().numpy())
        expected_radians = convert_lerobot_action_to_leisaac(expected_state)
        actual_radians = np.stack(replay_states)
        result = {
            "replay_kind": "action_replay_not_exact_episode_replay",
            "episode_index": args.episode_index,
            "fresh_reset_seed": args.seed,
            "action_rate_hz": args.action_rate_hz,
            "dataset_scene_state_available": False,
            "metrics_radians": trajectory_metrics(expected_radians, actual_radians),
            "success": "unknown: initial orange/plate states were not restored",
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = args.output_dir / f"episode_{args.episode_index:03d}_action_replay.json"
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Action replay result written to {path}")
    except Exception as exc:
        print(f"Action replay failed: {exc}", file=sys.stderr)
        return 3
    finally:
        if env is not None:
            env.close()
        app_launcher.app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
