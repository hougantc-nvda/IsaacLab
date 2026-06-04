# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression test for kitless XrAnchorManager construction.

In the kitless Newton XR path there is no Kit app, so carb's ``ISettings``
interface cannot be acquired and ``carb.settings.get_settings()`` raises a
``RuntimeError`` at runtime.  ``XrAnchorManager`` configures Kit's XR-render
settings, which the Newton renderer does not consume, so construction must
tolerate the unavailable interface instead of crashing the teleop session.

The reproduction runs in a subprocess so the stubbed ``carb`` / ``isaaclab.sim``
modules stay isolated and never clobber the real modules in a shared pytest
session (e.g. in CI where ``isaaclab.sim`` is importable).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

# Stub carb so ``carb.settings.get_settings()`` raises exactly like the kitless
# Newton path, stub the ``isaaclab.sim`` subtree so importing the manager does
# not boot the Isaac Sim kit kernel, then construct the manager and assert it
# survives the unavailable settings interface.
_KITLESS_CARB_REPRO = textwrap.dedent(
    """
    import sys, types

    carb = types.ModuleType("carb")
    settings_mod = types.ModuleType("carb.settings")
    def get_settings():
        raise RuntimeError(
            "Failed to acquire interface: carb::settings::ISettings (pluginName: nullptr)"
        )
    settings_mod.get_settings = get_settings
    carb.settings = settings_mod
    sys.modules["carb"] = carb
    sys.modules["carb.settings"] = settings_mod

    import isaaclab  # real package; only the isaaclab.sim subtree is stubbed below
    sim = types.ModuleType("isaaclab.sim")
    sim.SimulationContext = type("SimulationContext", (), {})
    sim_utils = types.ModuleType("isaaclab.sim.utils")
    sim_prims = types.ModuleType("isaaclab.sim.utils.prims")
    sim_prims.create_prim = lambda *a, **k: None
    sim_stage = types.ModuleType("isaaclab.sim.utils.stage")
    sim_stage.get_current_stage = lambda *a, **k: None
    sim_stage.get_current_stage_id = lambda *a, **k: -1
    sim.utils = sim_utils
    sim_utils.prims = sim_prims
    sim_utils.stage = sim_stage
    isaaclab.sim = sim
    sys.modules["isaaclab.sim"] = sim
    sys.modules["isaaclab.sim.utils"] = sim_utils
    sys.modules["isaaclab.sim.utils.prims"] = sim_prims
    sys.modules["isaaclab.sim.utils.stage"] = sim_stage

    from isaaclab_teleop.xr_anchor_manager import XrAnchorManager
    from isaaclab_teleop.xr_cfg import XrCfg

    manager = XrAnchorManager(
        XrCfg(anchor_pos=(1.0, 2.0, 3.0), anchor_rot=(0.0, 0.0, 0.0, 1.0), near_plane=0.2)
    )
    assert manager.anchor_headset_path == "/World/XRAnchor"
    print("XR_ANCHOR_MANAGER_OK")
    """
)


_KITLESS_DEVICE_REPRO = textwrap.dedent(
    """
    import sys, types

    carb = types.ModuleType("carb")
    settings_mod = types.ModuleType("carb.settings")
    def get_settings():
        raise RuntimeError(
            "Failed to acquire interface: carb::settings::ISettings (pluginName: nullptr)"
        )
    settings_mod.get_settings = get_settings
    carb.settings = settings_mod
    sys.modules["carb"] = carb
    sys.modules["carb.settings"] = settings_mod

    eventdispatcher = types.ModuleType("carb.eventdispatcher")
    def get_eventdispatcher():
        raise AssertionError("renderer-owned OpenXR path must not touch Kit pre-shutdown events")
    eventdispatcher.get_eventdispatcher = get_eventdispatcher
    sys.modules["carb.eventdispatcher"] = eventdispatcher

    omni = types.ModuleType("omni")
    kit = types.ModuleType("omni.kit")
    app = types.ModuleType("omni.kit.app")
    app.GLOBAL_EVENT_PRE_SHUTDOWN = "pre_shutdown"
    kit.app = app
    omni.kit = kit
    sys.modules["omni"] = omni
    sys.modules["omni.kit"] = kit
    sys.modules["omni.kit.app"] = app

    from isaaclab_teleop import IsaacTeleopCfg
    from isaaclab_teleop.isaac_teleop_device import create_isaac_teleop_device

    cfg = IsaacTeleopCfg(
        pipeline_builder=lambda: None,
        control_channel_uuid=None,
        openxr_handles_provider=lambda: None,
    )
    device = create_isaac_teleop_device(cfg, auto_launch_cloudxr=False)
    assert device._anchor_manager is None
    print("KITLESS_TELEOP_DEVICE_OK")
    """
)


def test_xr_anchor_manager_survives_unavailable_carb_settings():
    """Constructing the manager must not crash when carb settings cannot be acquired."""
    result = subprocess.run(
        [sys.executable, "-c", _KITLESS_CARB_REPRO],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"XrAnchorManager construction crashed:\n{result.stderr}"
    assert "XR_ANCHOR_MANAGER_OK" in result.stdout


def test_renderer_owned_openxr_device_skips_kit_anchor_manager_and_settings():
    """Renderer-owned OpenXR handles must not construct Kit-backed anchor helpers."""
    result = subprocess.run(
        [sys.executable, "-c", _KITLESS_DEVICE_REPRO],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Kitless teleop device construction crashed:\n{result.stderr}"
    assert "KITLESS_TELEOP_DEVICE_OK" in result.stdout
