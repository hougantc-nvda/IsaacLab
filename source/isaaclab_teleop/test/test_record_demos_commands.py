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


@pytest.mark.parametrize("backend", ["newton_mjwarp", "ovphysx"])
def test_recording_rejects_unsupported_task_action_policy(monkeypatch, tmp_path, backend):
    """Prepared humanoid policies must not reach an HDF5 loop that cannot apply them."""
    from isaaclab_teleop.entrypoints import record_demos

    monkeypatch.setattr(sys, "argv", ["record_demos.py"])
    record_demos._parse_args(
        ["--task", "IsaacContrib-PickPlace-GR1T2-Abs", "--physics", backend, "--renderer", "ovrtx"]
    )
    with pytest.raises(ValueError, match="does not run this task's kitless action policy"):
        record_demos.create_environment_config(str(tmp_path), "rejected")


def test_recording_prepares_so101_calibrated_reset(monkeypatch, tmp_path):
    from isaaclab_teleop.entrypoints import record_demos

    monkeypatch.setattr(sys, "argv", ["record_demos.py"])
    record_demos._parse_args(["--task", "IsaacContrib-Stack-Cube-SO101-IK-Abs-v0", "--physics", "isaacsim_physx"])
    cfg, _, _ = record_demos.create_environment_config(str(tmp_path), "stack")
    assert cfg.events.randomize_joint_state.params["std"] == 0.0


def test_recording_reset_notifies_teleop_after_environment_reset(monkeypatch):
    """An operator RESET already consumed by the control channel still resets retargeters."""
    from types import SimpleNamespace

    from isaaclab_teleop.entrypoints import record_demos

    calls = []
    monkeypatch.setattr(record_demos, "args_cli", SimpleNamespace(reset_sim_buffer_each_episode=False), raising=False)
    env = SimpleNamespace(
        recorder_manager=SimpleNamespace(reset=lambda: calls.append("recorder-reset")),
        reset=lambda: calls.append("env-reset"),
    )
    device = SimpleNamespace(
        reset=lambda: calls.append("device-reset"), notify_reset=lambda: calls.append("device-notify-reset")
    )

    assert record_demos.handle_reset(env, 3, SimpleNamespace(show_demo=lambda _text: None), "label", device) == 0
    assert calls == ["recorder-reset", "env-reset", "device-reset", "device-notify-reset"]


@pytest.mark.parametrize(
    ("terminated", "truncated", "notified"),
    [
        ((True, False), (False, False), True),
        ((False, False), (False, True), False),
        ((False, False), (True, False), True),
    ],
)
def test_recording_notifies_teleop_when_environment_zero_resets_itself(terminated, truncated, notified):
    """A termination inside ``env.step`` restarts teleop input only for the teleoperated environment."""
    from types import SimpleNamespace

    import torch
    from isaaclab_teleop.entrypoints import record_demos

    calls = []
    device = SimpleNamespace(notify_reset=lambda: calls.append("notify"))
    step_result = (None, None, torch.tensor(terminated), torch.tensor(truncated), {})
    assert record_demos.notify_teleop_of_auto_reset(step_result, device) is notified
    assert calls == (["notify"] if notified else [])
    # Devices without the notification keep their state, as before.
    assert record_demos.notify_teleop_of_auto_reset(step_result, SimpleNamespace()) is False
