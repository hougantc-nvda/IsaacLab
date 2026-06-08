# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for kitless teleoperation script imports."""

import ast
from pathlib import Path

_TELEOP_SCRIPT = (
    Path(__file__).resolve().parents[3] / "scripts" / "environments" / "teleoperation" / "teleop_se3_agent.py"
)


def test_teleop_se3_agent_keeps_legacy_devices_lazy() -> None:
    """Ensure the kitless script does not import Kit-backed devices at module import time."""
    tree = ast.parse(_TELEOP_SCRIPT.read_text())
    blocked_modules = {
        "isaaclab.devices",
        "isaaclab.devices.teleop_device_factory",
    }

    eager_imports = [
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.module in blocked_modules
    ]

    assert eager_imports == []
