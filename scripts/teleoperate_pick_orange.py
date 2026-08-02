#!/usr/bin/env python3
"""Teleoperate the frozen current-stack PickOrange scene with official LeIsaac devices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teleop-device", choices=["keyboard", "gamepad"], default="keyboard")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sensitivity", type=float, default=1.0)
    parser.add_argument(
        "--frozen-config", type=Path, default=Path("configs/pick_orange_current_stack.yaml")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        from isaaclab.app import AppLauncher
    except ImportError:
        print(
            "Teleoperation requires the installed Isaac Sim, Isaac Lab and LeIsaac stack.",
            file=sys.stderr,
        )
        return 3
    parser = build_parser()
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(enable_cameras=True, headless=False)
    args = parser.parse_args(argv)
    frozen = yaml.safe_load(args.frozen_config.read_text(encoding="utf-8"))
    app = AppLauncher(args).app
    env = None
    try:
        import gymnasium as gym
        import leisaac  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg

        env_cfg = parse_env_cfg(frozen["environment_id"], device=args.device, num_envs=1)
        env_cfg.use_teleop_device(args.teleop_device)
        env_cfg.seed = args.seed
        for name in vars(env_cfg.events):
            if name.startswith("domain_randomize_"):
                setattr(env_cfg.events, name, None)
        env = gym.make(frozen["environment_id"], cfg=env_cfg).unwrapped
        if args.teleop_device == "keyboard":
            from leisaac.devices import SO101Keyboard

            controller = SO101Keyboard(env, sensitivity=args.sensitivity)
        else:
            from leisaac.devices import SO101Gamepad

            controller = SO101Gamepad(env, sensitivity=args.sensitivity)
        controller.add_callback("R", env.reset)
        controller.display_controls()
        print("R: reset. Device-specific movement/gripper controls are printed above by LeIsaac.")
        env.reset()
        controller.reset()
        while app.is_running():
            action = controller.advance()
            if action is None:
                env.render()
            else:
                env.step(action)
    except Exception as exc:
        print(f"Teleoperation failed: {exc}", file=sys.stderr)
        return 3
    finally:
        if env is not None:
            env.close()
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
