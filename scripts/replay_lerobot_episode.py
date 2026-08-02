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


def load_episode_numeric_trajectory(
    dataset: object, episode_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Load one v3 episode's numeric columns without decoding its camera videos."""

    metadata = dataset.meta.episodes[episode_index]
    start = int(metadata["dataset_from_index"])
    stop = int(metadata["dataset_to_index"])
    frames = dataset.hf_dataset.select(range(start, stop))
    episode_ids = {int(value) for value in frames["episode_index"]}
    if episode_ids != {episode_index}:
        raise ValueError(
            f"Dataset index range [{start}, {stop}) does not exclusively contain episode {episode_index}."
        )
    actions = np.stack([value.detach().cpu().numpy() for value in frames["action"]])
    states = np.stack([value.detach().cpu().numpy() for value in frames["observation.state"]])
    return actions, states


def save_trajectory_plot(expected: np.ndarray, actual: np.ndarray, path: Path) -> None:
    """Save a compact per-joint expected-versus-replayed trajectory chart."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    joint_names = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )
    figure, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    for index, axis in enumerate(axes.flat):
        axis.plot(expected[:, index], label="dataset converted", linewidth=1.2)
        axis.plot(actual[:, index], label="replay", linewidth=1.0, alpha=0.8)
        axis.set_title(joint_names[index])
        axis.set_ylabel("radians")
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    axes[-1, 0].set_xlabel("dataset frame (30 Hz)")
    axes[-1, 1].set_xlabel("dataset frame (30 Hz)")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


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

        print(f"Loading dataset episode {args.episode_index}...", flush=True)
        dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root)
        if dataset.fps != args.action_rate_hz:
            print(
                f"Dataset metadata is {dataset.fps} Hz but replay was requested at {args.action_rate_hz} Hz. "
                "No implicit resampling is performed.",
                file=sys.stderr,
            )
            return 2
        actions, expected_state = load_episode_numeric_trajectory(dataset, args.episode_index)
        print(f"Loaded {len(actions)} frames at {dataset.fps} Hz.", flush=True)
        env_cfg = parse_env_cfg(args.task, device="cuda", num_envs=1)
        env_cfg.use_teleop_device("so101leader")
        dataset_duration_seconds = len(actions) / args.action_rate_hz
        configured_episode_length_seconds = env_cfg.episode_length_s
        control_rate_hz = 1.0 / (env_cfg.sim.dt * env_cfg.decimation)
        replay_episode_length_seconds = max(
            configured_episode_length_seconds,
            dataset_duration_seconds + 1.0 / control_rate_hz,
        )
        env_cfg.episode_length_s = replay_episode_length_seconds
        print(f"Creating {args.task}...", flush=True)
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        action_repeat = int(round(control_rate_hz / args.action_rate_hz))
        if action_repeat < 1 or not np.isclose(
            control_rate_hz, args.action_rate_hz * action_repeat
        ):
            raise ValueError(
                f"Environment control rate {control_rate_hz:g} Hz is not an integer multiple of "
                f"the dataset rate {args.action_rate_hz:g} Hz; no implicit resampling is allowed."
            )
        env.seed(args.seed)
        print(f"Resetting with fresh seed {args.seed}...", flush=True)
        env.reset()
        print(
            "Replaying actions after explicit LeRobot-to-radian conversion "
            f"({action_repeat} simulation steps per dataset action)...",
            flush=True,
        )
        replay_states: list[np.ndarray] = []
        for frame_index, action in enumerate(actions):
            radians = convert_lerobot_action_to_leisaac(action[None, :])
            env_action = torch.as_tensor(radians, dtype=torch.float32, device=env.device)
            for _ in range(action_repeat):
                _, _, terminated, truncated, _ = env.step(env_action)
                if bool(torch.any(terminated | truncated)):
                    raise RuntimeError(
                        f"Environment terminated during replay at dataset frame {frame_index}; "
                        "the trajectory comparison would be invalid."
                    )
            replay_states.append(env.scene["robot"].data.joint_pos[0, :6].detach().cpu().numpy())
            if frame_index and frame_index % 100 == 0:
                print(f"Replayed {frame_index}/{len(actions)} frames...", flush=True)
        expected_radians = convert_lerobot_action_to_leisaac(expected_state)
        actual_radians = np.stack(replay_states)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        trajectory_path = args.output_dir / f"episode_{args.episode_index:03d}_trajectories.npz"
        plot_path = args.output_dir / f"episode_{args.episode_index:03d}_trajectory_comparison.png"
        np.savez_compressed(
            trajectory_path,
            expected_radians=expected_radians,
            replayed_radians=actual_radians,
        )
        save_trajectory_plot(expected_radians, actual_radians, plot_path)
        result = {
            "replay_kind": "action_replay_not_exact_episode_replay",
            "episode_index": args.episode_index,
            "fresh_reset_seed": args.seed,
            "action_rate_hz": args.action_rate_hz,
            "environment_control_rate_hz": control_rate_hz,
            "simulation_steps_per_dataset_action": action_repeat,
            "configured_episode_length_seconds": configured_episode_length_seconds,
            "replay_episode_length_seconds": replay_episode_length_seconds,
            "dataset_scene_state_available": False,
            "trajectory_data": str(trajectory_path),
            "trajectory_plot": str(plot_path),
            "metrics_radians": trajectory_metrics(expected_radians, actual_radians),
            "success": "unknown: initial orange/plate states were not restored",
        }
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
