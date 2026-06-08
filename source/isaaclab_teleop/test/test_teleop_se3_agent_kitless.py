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


def _read_teleop_script_tree() -> ast.Module:
    return ast.parse(_TELEOP_SCRIPT.read_text())


def test_teleop_se3_agent_keeps_legacy_devices_lazy() -> None:
    """Ensure the kitless script does not import Kit-backed devices at module import time."""
    tree = _read_teleop_script_tree()
    blocked_modules = {
        "isaaclab.devices",
        "isaaclab.devices.teleop_device_factory",
    }

    eager_imports = [
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.module in blocked_modules
    ]

    assert eager_imports == []


def test_teleop_se3_agent_defers_cloudxr_launch_until_after_env_creation() -> None:
    """Ensure CloudXR does not run concurrently with Newton USD import during env creation."""
    tree = _read_teleop_script_tree()
    main_func = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")

    env_creation_lines = [
        node.lineno
        for node in ast.walk(main_func)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "gym"
        and node.func.attr == "make"
    ]
    cloudxr_launch_lines = [
        node.lineno
        for node in ast.walk(main_func)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "launch_cloudxr"
    ]

    assert env_creation_lines
    assert cloudxr_launch_lines
    assert min(cloudxr_launch_lines) > max(env_creation_lines)
