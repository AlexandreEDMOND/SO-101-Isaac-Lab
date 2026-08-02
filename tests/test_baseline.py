from __future__ import annotations

from pathlib import Path

import yaml

from so101_sorting.baseline import (
    EXPECTED_JOINT_NAMES,
    parse_lerobot_training_metrics,
    summarize_episode_lengths,
    validate_act_config,
    validate_pick_orange_metadata,
)


def _metadata() -> dict[str, object]:
    vector = {"dtype": "float32", "shape": [6], "names": list(EXPECTED_JOINT_NAMES)}
    return {
        "codebase_version": "v3.0",
        "fps": 30,
        "features": {
            "observation.state": vector,
            "action": vector,
            "observation.images.front": {"dtype": "video", "shape": [480, 640, 3]},
            "observation.images.wrist": {"dtype": "video", "shape": [480, 640, 3]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }


def test_pick_orange_metadata_validation_accepts_expected_features() -> None:
    result = validate_pick_orange_metadata(_metadata())
    assert result.is_compatible_schema
    assert not result.warnings


def test_pick_orange_metadata_validation_rejects_missing_wrist_camera() -> None:
    metadata = _metadata()
    features = metadata["features"]
    assert isinstance(features, dict)
    del features["observation.images.wrist"]
    result = validate_pick_orange_metadata(metadata)
    assert "Missing required camera feature: observation.images.wrist." in result.errors


def test_episode_duration_summary() -> None:
    summary = summarize_episode_lengths([30, 60, 90], fps=30)
    assert summary["min_seconds"] == 1
    assert summary["mean_seconds"] == 2
    assert summary["max_seconds"] == 3


def test_parse_lerobot_metrics() -> None:
    metrics = parse_lerobot_training_metrics(
        ["step: 5 smpl: 10 ep: 0 epch: 0.1 loss: 1.50 grdn: 2.0 lr: 1e-05"]
    )
    assert metrics == [{"step": 5, "loss": 1.5, "learning_rate": 1e-05}]


def test_act_config_is_valid_for_the_intended_subset() -> None:
    config = yaml.safe_load(Path("configs/act_pick_orange_smoke.yaml").read_text(encoding="utf-8"))
    assert validate_act_config(config) == []


def test_training_analysis_writes_outputs(tmp_path: Path) -> None:
    from scripts.analyze_training import analyze

    training_dir = tmp_path / "training"
    checkpoint = training_dir / "checkpoints/last/pretrained_model"
    checkpoint.mkdir(parents=True)
    (checkpoint / "train_config.json").write_text("{}", encoding="utf-8")
    log_file = tmp_path / "train.log"
    log_file.write_text(
        "# started_at=2026-08-02T10:00:00+00:00\n"
        "step: 10 loss: 0.75 lr: 1e-05\n"
        "# finished_at=2026-08-02T10:00:05+00:00\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "analysis"

    summary = analyze(training_dir, log_file, output_dir)

    assert summary["total_logged_steps"] == 10
    assert summary["duration_seconds"] == 5
    assert (output_dir / "training_loss.svg").is_file()
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "summary.md").is_file()
