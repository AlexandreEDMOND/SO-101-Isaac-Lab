"""Integration checks are opt-in: they require LeRobot, CUDA and remote data."""

from __future__ import annotations

import os

import pytest


@pytest.mark.integration
def test_pick_orange_v3_inspection_requires_explicit_environment() -> None:
    if os.environ.get("RUN_PICK_ORANGE_INTEGRATION") != "1":
        pytest.skip("Set RUN_PICK_ORANGE_INTEGRATION=1 in the pinned LeRobot/Isaac environment.")
    pytest.importorskip("lerobot")
    pytest.importorskip("torch")
