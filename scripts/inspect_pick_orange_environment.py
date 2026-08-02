#!/usr/bin/env python3
"""Generate a runtime manifest from an installed LeIsaac PickOrange environment.

This script intentionally imports Isaac Lab only after AppLauncher has started.
It does not apply a policy or download assets. It fails rather than filling a
missing runtime value from the historical compatibility YAML.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from so101_sorting.current_stack import environment_fingerprint

TASK_ID = "LeIsaac-SO101-PickOrange-v0"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=TASK_ID)
    parser.add_argument("--teleop-device", default="so101leader", choices=["so101leader"])
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/compatibility/environment_manifest.json")
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--disable-cameras",
        action="store_true",
        help="Runtime diagnostic only: start the non-rendering Isaac Lab experience.",
    )
    parser.add_argument(
        "--frozen-config", type=Path, default=Path("configs/pick_orange_current_stack.yaml")
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _node(value: Any, source: str, status: str = "confirmed", notes: str = "") -> dict[str, Any]:
    return {"value": value, "status": status, "source": source, "notes": notes}


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _camera_manifest(name: str, camera_cfg: Any, sensor: Any | None) -> dict[str, Any]:
    source = "installed LeIsaac environment configuration"
    resolution = [camera_cfg.height, camera_cfg.width, 3]
    if sensor is not None and hasattr(sensor, "image_shape"):
        height, width = sensor.image_shape
        resolution = [int(height), int(width), 3]
    spawn = camera_cfg.spawn
    offset = camera_cfg.offset
    return {
        "environment_name": _node(name, source),
        "prim_path": _node(camera_cfg.prim_path, source),
        "recorded_resolution": _node(resolution, source),
        "extrinsics_ros_wxyz": _node(
            {
                "position": list(offset.pos),
                "quaternion_wxyz": list(offset.rot),
                "convention": offset.convention,
            },
            source,
        ),
        "intrinsics": _node(
            {
                "focal_length_mm": spawn.focal_length,
                "horizontal_aperture_mm": spawn.horizontal_aperture,
                "focus_distance": spawn.focus_distance,
                "clipping_range_m": list(spawn.clipping_range),
            },
            source,
        ),
        "update_period_seconds": _node(camera_cfg.update_period, source),
    }


def inspect_environment(args: argparse.Namespace) -> dict[str, Any]:
    """Start Isaac Lab, construct the real Gym environment, and inspect its config/runtime handles."""

    env = None
    try:
        import gymnasium as gym
        import leisaac  # noqa: F401  # Registers the official environment ID.
        from isaaclab_tasks.utils import parse_env_cfg

        frozen = yaml.safe_load(args.frozen_config.read_text(encoding="utf-8"))
        if frozen["environment_id"] != args.task:
            raise RuntimeError("--task does not match the frozen current-stack configuration.")
        env_cfg = parse_env_cfg(args.task, device="cuda", num_envs=args.num_envs)
        env_cfg.use_teleop_device(args.teleop_device)
        for name in vars(env_cfg.events):
            if name.startswith("domain_randomize_"):
                setattr(env_cfg.events, name, None)
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        robot = env.scene["robot"]
        joint_names = list(robot.joint_names)
        selected_ids = [joint_names.index(name) for name in JOINTS]
        selected_limits = robot.data.joint_pos_limits[0, selected_ids].detach().cpu().tolist()
        asset_path = Path(env_cfg.scene.robot.spawn.usd_path)
        action_dim = int(env.action_manager.total_action_dim)
        sensors = env.scene.sensors
        errors: list[str] = []
        for name in ("front", "wrist"):
            if name not in sensors:
                errors.append(f"Missing camera sensor: {name}")
        if action_dim != 6:
            errors.append(f"Expected six leader joint actions, got {action_dim}.")
        if any(name not in joint_names for name in JOINTS):
            errors.append(f"Robot joint names do not contain {JOINTS!r}.")
        if errors:
            raise RuntimeError("; ".join(errors))
        sim_dt = float(env.cfg.sim.dt)
        decimation = int(env.cfg.decimation)
        return {
            "environment_fingerprint": environment_fingerprint(frozen),
            "frozen_config": str(args.frozen_config),
            "software": {
                "python": _node(sys.version.split()[0], "runtime"),
                "isaac_lab": _node(_version("isaaclab"), "installed distribution"),
                "leisaac": _node(_version("leisaac"), "installed distribution"),
                "lerobot": _node(_version("lerobot"), "installed distribution"),
                "isaac_sim": _node(_version("isaacsim"), "installed distribution"),
            },
            "environment": {"id": _node(args.task, "gym runtime")},
            "robot": {
                "type": _node(getattr(env_cfg, "robot_name", "unknown"), "environment config"),
                "asset_path": _node(str(asset_path), "environment config"),
                "asset_sha256": _node(
                    _sha256(asset_path),
                    "runtime filesystem",
                    "confirmed" if asset_path.is_file() else "unknown",
                ),
                "joint_order": _node(
                    [joint_names[index] for index in selected_ids], "runtime articulation"
                ),
                "joint_limits_radians": _node(selected_limits, "runtime articulation"),
                "initial_joint_position_radians": _node(
                    [env_cfg.scene.robot.init_state.joint_pos[name] for name in JOINTS],
                    "environment config",
                ),
                "actuators": _node(
                    {
                        key: {
                            "joint_names_expr": value.joint_names_expr,
                            "effort_limit_sim": value.effort_limit_sim,
                            "velocity_limit_sim": value.velocity_limit_sim,
                            "stiffness": value.stiffness,
                            "damping": value.damping,
                        }
                        for key, value in env_cfg.scene.robot.actuators.items()
                    },
                    "environment config",
                ),
            },
            "actions": {
                "dimension": _node(action_dim, "runtime action manager"),
                "joint_order": _node(
                    [f"{name}.pos" for name in JOINTS], "leader action config mapping"
                ),
                "command_type": _node("absolute_joint_position", "so101leader action config"),
                "units_before_dataset_conversion": _node("radians", "LeIsaac action processing"),
            },
            "state": {
                "dimension": _node(6, "runtime articulation mapping"),
                "joint_order": _node(
                    [f"{name}.pos" for name in JOINTS], "runtime articulation mapping"
                ),
                "units": _node("radians", "runtime articulation"),
            },
            "cameras": {
                "front": _camera_manifest("front", env_cfg.scene.front, sensors.get("front")),
                "wrist": _camera_manifest("wrist", env_cfg.scene.wrist, sensors.get("wrist")),
            },
            "simulation": {
                "physics_timestep_seconds": _node(sim_dt, "runtime environment config"),
                "physics_timestep_hz": _node(1.0 / sim_dt, "runtime environment config"),
                "decimation": _node(decimation, "runtime environment config"),
                "control_frequency_hz": _node(
                    1.0 / (sim_dt * decimation), "runtime environment config"
                ),
                "episode_length_seconds": _node(
                    env.cfg.episode_length_s, "runtime environment config"
                ),
                "randomization": _node(str(env_cfg.events), "runtime event configuration"),
            },
        }
    except Exception as exc:
        print(f"Environment construction failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    finally:
        if env is not None:
            env.close()


def main(argv: list[str] | None = None) -> int:
    try:
        from isaaclab.app import AppLauncher
    except ImportError:
        print(
            "Environment inspection failed: Isaac Lab is unavailable. Install the pinned stack.",
            file=sys.stderr,
        )
        return 3
    parser = build_parser()
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(enable_cameras=True, headless=True)
    args = parser.parse_args(argv)
    if args.disable_cameras:
        args.enable_cameras = False
    app_launcher = AppLauncher(args)
    try:
        manifest = inspect_environment(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"Environment manifest written to {args.output}")
        return 0
    except Exception as exc:
        print(f"Environment inspection failed: {exc}", file=sys.stderr)
        return 3
    finally:
        app_launcher.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
