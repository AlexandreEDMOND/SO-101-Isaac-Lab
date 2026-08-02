from __future__ import annotations

from scripts.inspect_dataset import parse_args as parse_inspect_args
from scripts.train_act_pick_orange import build_command, compatibility_report_allows_training
from scripts.train_act_pick_orange import parse_args as parse_train_args


def test_inspect_dataset_argument_parsing() -> None:
    args = parse_inspect_args(
        ["--repo-id", "org/data", "--episode-index", "4", "--no-download-videos"]
    )
    assert args.repo_id == "org/data"
    assert args.episode_index == 4
    assert args.no_download_videos


def test_train_command_is_dry_run_ready() -> None:
    args = parse_train_args(["--mode", "smoke", "--dataset-repo-id", "org/data-v3"])
    command, output_dir = build_command(args)
    assert "--dataset.repo_id=org/data-v3" in command
    assert output_dir.name == "act_pick_orange_smoke"


def test_train_command_accepts_a_local_v3_dataset() -> None:
    args = parse_train_args(
        ["--dataset-repo-id", "org/data-v3", "--dataset-root", "/data/pick-orange-v3"]
    )
    command, _ = build_command(args)
    assert "--dataset.root=/data/pick-orange-v3" in command


def test_resume_uses_checkpoint_configuration() -> None:
    args = parse_train_args(
        ["--mode", "full", "--dataset-repo-id", "org/data-v3", "--output-dir", "run", "--resume"]
    )
    command, _ = build_command(args)
    assert "--config_path=run/checkpoints/last/pretrained_model/train_config.json" in command
    assert "--resume=true" in command


def test_train_parser_accepts_compatibility_gate() -> None:
    args = parse_train_args(["--dataset-repo-id", "org/data-v3", "--require-compatibility-report"])
    assert args.require_compatibility_report


def test_compatibility_gate_rejects_blocking_failure(tmp_path) -> None:
    report = tmp_path / "report.json"
    report.write_text('{"blocking_failures": 1}', encoding="utf-8")
    allowed, message = compatibility_report_allows_training(report)
    assert not allowed
    assert "blocking FAIL" in message
