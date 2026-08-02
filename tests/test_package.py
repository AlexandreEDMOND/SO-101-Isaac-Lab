"""Tests that are intentionally independent from Isaac Sim."""

from so101_sorting import __version__


def test_package_exposes_a_version() -> None:
    """The lightweight package remains importable without the simulator."""

    assert __version__ == "0.1.0"
