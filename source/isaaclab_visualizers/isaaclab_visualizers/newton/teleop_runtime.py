# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-specific helpers for renderer-owned OpenXR teleoperation."""

from __future__ import annotations

from isaaclab_teleop.openxr_runtime import requested_preset_names, visualizer_types_from_args

NEWTON_PHYSICS_PRESETS = {"newton", "newton_mjwarp"}
"""Preset names that request Newton physics."""

NEWTON_RENDERER_PRESETS = {"newton_renderer"}
"""Preset names that request the Newton renderer."""

NEWTON_KITLESS_PRESETS = NEWTON_PHYSICS_PRESETS | NEWTON_RENDERER_PRESETS
"""Newton fallback preset names that may need stripping for tasks without preset declarations."""

NEWTON_OPENXR_RENDER_INTERVAL = 3
"""Default render interval for kitless Newton OpenXR teleoperation."""


def newton_openxr_visualizer_requested(launcher_args) -> bool:
    """Return whether the CLI requests Newton as an OpenXR visualizer."""
    return bool(getattr(launcher_args, "xr", False)) and "newton" in visualizer_types_from_args(launcher_args)


def newton_physics_requested(launcher_args, original_argv: list[str], current_argv: list[str]) -> bool:
    """Return whether the CLI requests Newton physics for kitless Newton XR."""
    return newton_openxr_visualizer_requested(launcher_args) and bool(
        requested_preset_names(original_argv, current_argv) & NEWTON_PHYSICS_PRESETS
    )


def env_cfg_uses_newton_physics(env_cfg) -> bool:
    """Return whether an environment config is using Newton physics."""
    try:
        from isaaclab_newton.physics import NewtonCfg
    except ImportError:
        return False
    return isinstance(getattr(getattr(env_cfg, "sim", None), "physics", None), NewtonCfg)


def env_uses_newton_physics(env) -> bool:
    """Return whether an environment is using Newton physics."""
    return env_cfg_uses_newton_physics(getattr(env, "cfg", None))


def apply_kitless_newton_backend(
    env_cfg, *, enabled: bool, render_interval: int = NEWTON_OPENXR_RENDER_INTERVAL
) -> None:
    """Apply Newton physics and task-owned Newton overrides for kitless XR."""
    if not enabled:
        return

    from isaaclab_newton.physics import NewtonCfg
    from isaaclab_newton.physics.mjwarp_manager_cfg import MJWarpSolverCfg

    physics_cfg = getattr(env_cfg.sim, "physics", None)
    if not isinstance(physics_cfg, NewtonCfg):
        physics_cfg = NewtonCfg(solver_cfg=MJWarpSolverCfg())
        env_cfg.sim.physics = physics_cfg

    solver_cfg = physics_cfg.solver_cfg
    if isinstance(solver_cfg, MJWarpSolverCfg):
        solver_cfg.nconmax = max(solver_cfg.nconmax or 0, 256)
        solver_cfg.njmax = max(solver_cfg.njmax or 0, 512)
    physics_cfg.num_substeps = max(physics_cfg.num_substeps, 2)
    env_cfg.decimation = render_interval
    env_cfg.sim.render_interval = render_interval

    kitless_newton_overrides = getattr(env_cfg, "apply_kitless_newton_overrides", None)
    if callable(kitless_newton_overrides):
        kitless_newton_overrides()


def default_newton_openxr_visualization_device(env_cfg, *, enabled: bool) -> None:
    """Default Newton OpenXR visualization to CUDA when physics is CPU."""
    if not enabled or env_cfg_uses_newton_physics(env_cfg):
        return
    if str(getattr(env_cfg.sim, "device", "cpu")) != "cpu":
        return
    try:
        import warp as wp
        from isaaclab_newton.physics import NewtonManager
    except ImportError:
        return
    if NewtonManager.get_visualization_device_override():
        return
    if wp.is_cuda_available():
        NewtonManager.set_visualization_device("cuda:0")
        print("Renderer-owned XR will use Newton visualization device cuda:0 with CPU physics.")


def log_newton_openxr_backend_selection(env_cfg, *, enabled: bool) -> None:
    """Log backend selection for Newton OpenXR runs."""
    if not enabled:
        return

    physics_cfg = getattr(getattr(env_cfg, "sim", None), "physics", None)
    physics_name = type(physics_cfg).__name__ if physics_cfg is not None else "None"
    sim_device = getattr(env_cfg.sim, "device", None)
    newton_physics_active = env_cfg_uses_newton_physics(env_cfg)
    try:
        from isaaclab_newton.physics import NewtonManager

        visualization_device = NewtonManager.get_visualization_device_override() or "<sim_device>"
    except ImportError:
        visualization_device = "<sim_device>"
    print(
        "Renderer-owned XR backend selection: "
        f"physics={physics_name}, sim_device={sim_device}, "
        f"newton_visualization_device={visualization_device}, "
        f"newton_physics_active={newton_physics_active}."
    )


class NewtonOpenXRTeleopRuntime:
    """Newton implementation of the renderer-owned OpenXR teleop runtime contract."""

    visualizer_type = "newton"
    display_name = "Newton"
    cloudxr_label = "renderer-owned Newton XR"
    loader_required = True
    fallback_presets = NEWTON_KITLESS_PRESETS
    render_interval = NEWTON_OPENXR_RENDER_INTERVAL

    def visualizer_requested(self, launcher_args) -> bool:
        """Return whether the CLI requests Newton as an OpenXR visualizer."""
        return newton_openxr_visualizer_requested(launcher_args)

    def physics_requested(self, launcher_args, original_argv: list[str], current_argv: list[str]) -> bool:
        """Return whether the CLI requests Newton physics for kitless Newton XR."""
        return newton_physics_requested(launcher_args, original_argv, current_argv)

    def env_cfg_uses_runtime_physics(self, env_cfg) -> bool:
        """Return whether an environment config is using Newton physics."""
        return env_cfg_uses_newton_physics(env_cfg)

    def env_uses_runtime_physics(self, env) -> bool:
        """Return whether an environment is using Newton physics."""
        return env_uses_newton_physics(env)

    def apply_backend(self, env_cfg, *, enabled: bool) -> None:
        """Apply Newton physics and task-owned Newton overrides for kitless XR."""
        apply_kitless_newton_backend(env_cfg, enabled=enabled, render_interval=self.render_interval)

    def default_visualization_device(self, env_cfg, *, enabled: bool) -> None:
        """Default Newton OpenXR visualization to CUDA when physics is CPU."""
        default_newton_openxr_visualization_device(env_cfg, enabled=enabled)

    def log_backend_selection(self, env_cfg, *, enabled: bool) -> None:
        """Log backend selection for Newton OpenXR runs."""
        log_newton_openxr_backend_selection(env_cfg, enabled=enabled)

    def debug_idle_enabled(self) -> bool:
        """Return whether to run kitless Newton with synthetic idle actions and no client."""
        return False

    def debug_step_limit(self, logger=None) -> int | None:
        """Return the synthetic idle-action debug step limit, if configured."""
        return None


_NEWTON_OPENXR_TELEOP_RUNTIME = NewtonOpenXRTeleopRuntime()


def get_openxr_teleop_runtime() -> NewtonOpenXRTeleopRuntime:
    """Return the Newton renderer-owned OpenXR teleop runtime policy."""
    return _NEWTON_OPENXR_TELEOP_RUNTIME
