# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression coverage for recording command configuration."""

import math
import sys

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "backend_flags", [["--physics", "isaacsim_physx"], ["--physics", "physx", "--xr"], ["--physics", "newton_mjwarp"]]
)
def test_recording_defers_command_resampling(monkeypatch, tmp_path, backend_flags):
    """Recording defers command resampling beyond a realistic demonstration."""
    from isaaclab_teleop.entrypoints import record_demos

    monkeypatch.setattr(sys, "argv", ["record_demos.py"])
    record_demos._parse_args(["--task", "Isaac-Reach-Franka", *backend_flags, "--teleop_device", "spacemouse"])
    cfg, _, _ = record_demos.create_environment_config(str(tmp_path), "reach")

    resampling_time_range = cfg.commands.ee_pose.resampling_time_range
    assert all(math.isfinite(bound) for bound in resampling_time_range)
    assert min(resampling_time_range) > 24 * 60 * 60


def test_recording_prepares_so101_calibrated_reset(monkeypatch, tmp_path):
    from isaaclab_teleop.entrypoints import record_demos

    monkeypatch.setattr(sys, "argv", ["record_demos.py"])
    record_demos._parse_args(["--task", "IsaacContrib-Stack-Cube-SO101-IK-Abs-v0", "--physics", "isaacsim_physx"])
    cfg, _, _ = record_demos.create_environment_config(str(tmp_path), "stack")
    assert cfg.events.randomize_joint_state.params["std"] == 0.0
