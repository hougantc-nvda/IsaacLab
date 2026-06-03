# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for non-Kit OpenXR teleoperation runtimes."""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

_RENDERER_OPENXR_RUNTIME_MODULES: dict[str, str] = {}


class RendererOpenXRTeleopRuntime(Protocol):
    """Renderer-specific policy for a renderer-owned OpenXR teleop path."""

    visualizer_type: str
    """Visualizer type name that activates this runtime."""

    display_name: str
    """Human-readable renderer name used in diagnostics."""

    cloudxr_label: str
    """Human-readable CloudXR launch label."""

    loader_required: bool
    """Whether this runtime needs the packaged OpenXR loader on ``LD_LIBRARY_PATH``."""

    fallback_presets: set[str]
    """Preset names that may be stripped when a task has no matching preset declarations."""

    render_interval: int
    """Preferred render interval for renderer-owned XR teleop."""

    def visualizer_requested(self, launcher_args) -> bool:
        """Return whether ``launcher_args`` request this OpenXR visualizer."""
        ...

    def physics_requested(self, launcher_args, original_argv: list[str], current_argv: list[str]) -> bool:
        """Return whether ``launcher_args`` request this runtime's physics backend."""
        ...

    def env_cfg_uses_runtime_physics(self, env_cfg) -> bool:
        """Return whether an environment config uses this runtime's physics backend."""
        ...

    def env_uses_runtime_physics(self, env) -> bool:
        """Return whether an environment uses this runtime's physics backend."""
        ...

    def apply_backend(self, env_cfg, *, enabled: bool) -> None:
        """Apply this runtime's physics/backend overrides when ``enabled`` is true."""
        ...

    def default_visualization_device(self, env_cfg, *, enabled: bool) -> None:
        """Apply this runtime's visualization-device defaults when ``enabled`` is true."""
        ...

    def log_backend_selection(self, env_cfg, *, enabled: bool) -> None:
        """Log this runtime's backend selection when ``enabled`` is true."""
        ...

    def debug_idle_enabled(self) -> bool:
        """Return whether this runtime should use a synthetic idle teleop device."""
        ...

    def debug_step_limit(self, logger: logging.Logger | None = None) -> int | None:
        """Return the synthetic idle-action debug step limit, if configured."""
        ...


def register_renderer_openxr_runtime_module(visualizer_type: str, module_name: str) -> None:
    """Register a renderer-owned OpenXR teleop runtime module.

    Args:
        visualizer_type: Visualizer type name used by ``--visualizer``.
        module_name: Import path exposing ``get_openxr_teleop_runtime()``.
    """
    _RENDERER_OPENXR_RUNTIME_MODULES[visualizer_type.strip().lower()] = module_name


def visualizer_types_from_args(launcher_args) -> set[str]:
    """Return normalized visualizer type names from parsed launcher arguments."""
    visualizers = getattr(launcher_args, "visualizer", None)
    if visualizers is None:
        return set()
    if isinstance(visualizers, str):
        visualizers = [token.strip() for token in visualizers.split(",")]
    return {str(visualizer).strip().lower() for visualizer in visualizers if str(visualizer).strip()}


def renderer_openxr_runtime_for_visualizer(visualizer_type: str) -> RendererOpenXRTeleopRuntime | None:
    """Return the renderer-owned OpenXR runtime registered for a visualizer type.

    Runtime modules are discovered by convention as
    ``isaaclab_visualizers.<visualizer_type>.teleop_runtime`` unless an explicit
    override was registered with :func:`register_renderer_openxr_runtime_module`.
    """
    normalized_type = visualizer_type.strip().lower()
    module_name = _RENDERER_OPENXR_RUNTIME_MODULES.get(
        normalized_type, f"isaaclab_visualizers.{normalized_type}.teleop_runtime"
    )

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name is not None and module_name.startswith(exc.name):
            return None
        raise

    factory = getattr(module, "get_openxr_teleop_runtime", None)
    if not callable(factory):
        raise RuntimeError(f"{module_name} must expose get_openxr_teleop_runtime().")
    return factory()


def renderer_openxr_runtime_for_args(launcher_args) -> RendererOpenXRTeleopRuntime | None:
    """Return the renderer-owned OpenXR runtime requested by parsed launcher arguments."""
    if not bool(getattr(launcher_args, "xr", False)):
        return None

    visualizers = visualizer_types_from_args(launcher_args)
    if "kit" in visualizers:
        return None

    for visualizer_type in sorted(visualizers):
        runtime = renderer_openxr_runtime_for_visualizer(visualizer_type)
        if runtime is not None and runtime.visualizer_requested(launcher_args):
            return runtime
    return None


def renderer_owned_openxr_requested(
    launcher_args,
    renderer_types: set[str] | tuple[str, ...] | None = None,
    runtime: RendererOpenXRTeleopRuntime | None = None,
) -> bool:
    """Return whether OpenXR should be owned by a non-Kit renderer visualizer.

    Args:
        launcher_args: Parsed launcher CLI arguments.
        renderer_types: Optional visualizer type names that can own OpenXR.
        runtime: Optional pre-resolved renderer-owned OpenXR runtime.

    Returns:
        ``True`` when XR is requested with a registered renderer and without Kit.
    """
    if renderer_types is None:
        return runtime is not None or renderer_openxr_runtime_for_args(launcher_args) is not None

    visualizers = visualizer_types_from_args(launcher_args)
    renderer_type_set = {str(renderer_type).lower() for renderer_type in renderer_types}
    return (
        bool(getattr(launcher_args, "xr", False)) and bool(visualizers & renderer_type_set) and "kit" not in visualizers
    )


def launcher_args_for_renderer_owned_openxr(
    launcher_args: argparse.Namespace,
    renderer_types: set[str] | tuple[str, ...] | None = None,
    runtime: RendererOpenXRTeleopRuntime | None = None,
) -> argparse.Namespace:
    """Return launcher args with Kit OpenXR disabled when a renderer owns OpenXR.

    Args:
        launcher_args: Parsed launcher CLI arguments.
        renderer_types: Optional visualizer type names that can own OpenXR.
        runtime: Optional pre-resolved renderer-owned OpenXR runtime.

    Returns:
        The original namespace, or a shallow copy with ``xr`` set to ``False``.
    """
    if not renderer_owned_openxr_requested(launcher_args, renderer_types, runtime):
        return launcher_args
    copied_args = argparse.Namespace(**vars(launcher_args))
    copied_args.xr = False
    return copied_args


def requested_preset_names(original_argv: list[str], current_argv: list[str]) -> set[str]:
    """Return preset names requested before or after launcher CLI folding."""
    preset_names: set[str] = set()
    for token in [*original_argv[1:], *current_argv[1:]]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key.lstrip("+") != "presets":
            continue
        preset_names.update(name.strip() for name in value.split(",") if name.strip())
    return preset_names


def strip_preset_tokens(tokens: list[str], presets_to_strip: set[str]) -> list[str]:
    """Remove selected preset names from CLI override tokens."""
    stripped: list[str] = []
    for token in tokens:
        if "=" not in token:
            stripped.append(token)
            continue
        key, value = token.split("=", 1)
        if key != "presets":
            stripped.append(token)
            continue
        names = [name.strip() for name in value.split(",") if name.strip()]
        remaining = [name for name in names if name not in presets_to_strip]
        if remaining:
            stripped.append(f"presets={','.join(remaining)}")
    return stripped


def apply_renderer_owned_teleop_overrides(env_cfg, *, render_interval: int = 3) -> None:
    """Apply generic config overrides needed by renderer-owned OpenXR teleop.

    Args:
        env_cfg: Isaac Lab environment configuration.
        render_interval: Minimum render interval used for renderer-owned XR.
    """
    env_cfg.sim.render_interval = max(env_cfg.sim.render_interval, render_interval)
    kitless_teleop_overrides = getattr(env_cfg, "apply_kitless_teleop_overrides", None)
    if callable(kitless_teleop_overrides):
        kitless_teleop_overrides()


def find_packaged_openxr_loader_dir(package_name: str = "isaacteleop") -> str | None:
    """Find a packaged OpenXR loader directory inside installed Python packages.

    Args:
        package_name: Python package tree to scan for ``libopenxr_loader``.

    Returns:
        Directory containing the packaged loader, or ``None`` when not found.
    """
    import site

    loader_names = ("libopenxr_loader.so.1", "libopenxr_loader.so")
    package_roots = [Path(site_dir) / package_name for site_dir in site.getsitepackages()]
    for loader_name in loader_names:
        for package_root in package_roots:
            if not package_root.is_dir():
                continue
            for candidate in sorted(package_root.rglob(loader_name)):
                return str(candidate.parent)

    return None


def reexec_with_packaged_openxr_loader_if_needed(
    *,
    enabled: bool,
    argv: list[str],
    package_name: str = "isaacteleop",
) -> None:
    """Restart the process once with a packaged OpenXR loader on ``LD_LIBRARY_PATH``.

    Args:
        enabled: Whether the current launch path needs a non-Kit OpenXR loader.
        argv: Original process arguments to preserve across re-exec.
        package_name: Python package tree that may contain the OpenXR loader.
    """
    if not enabled:
        return

    loader_dir = find_packaged_openxr_loader_dir(package_name)
    if loader_dir is None:
        return

    old_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    current_paths = [path for path in old_ld_library_path.split(":") if path]
    if loader_dir in current_paths:
        return

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{loader_dir}:{old_ld_library_path}" if old_ld_library_path else loader_dir
    os.execvpe(sys.executable, [sys.executable, *argv], env)


def resolve_cloudxr_env(value: str | None) -> str | None:
    """Resolve CloudXR profile shorthands to absolute ``.env`` file paths.

    Args:
        value: ``"cloudxrjs"``, ``"avp"``, ``"newton"``, ``"none"``, ``None``, or a file path.

    Returns:
        Resolved profile path, or ``None`` when CloudXR launch is disabled.
    """
    if value is None:
        return None

    value = value.strip()
    if value == "" or value.lower() == "none":
        return None

    from .isaac_teleop_cfg import CLOUDXR_ENV_PROFILES

    return CLOUDXR_ENV_PROFILES.get(value.lower(), value)


def launch_cloudxr_runtime(
    cloudxr_env_path: str | None,
    *,
    auto_launch: bool,
    label: str,
) -> object | None:
    """Launch the CloudXR runtime for a non-Kit OpenXR renderer.

    Args:
        cloudxr_env_path: Device/runtime CloudXR profile path, or ``None`` to disable launch.
        auto_launch: Whether launch is enabled by CLI/config.
        label: Human-readable runtime label for logs.

    Returns:
        The ``CloudXRLauncher`` instance, or ``None`` when launch is skipped.
    """
    if cloudxr_env_path is None or not auto_launch:
        return None
    from isaacteleop.cloudxr import CloudXRLauncher

    print(f"Starting CloudXR runtime for {label}: {cloudxr_env_path}")
    launcher = CloudXRLauncher(
        install_dir=str(Path.home() / ".cloudxr"),
        env_config=cloudxr_env_path,
        accept_eula=False,
    )
    logger.info("CloudXR runtime launched for %s.", label)
    return launcher


def configure_renderer_openxr_teleop(env, env_cfg) -> object | None:
    """Configure the active visualizer that owns a non-Kit OpenXR session.

    Args:
        env: Isaac Lab environment with initialized visualizers.
        env_cfg: Isaac Lab environment configuration containing ``isaac_teleop``.

    Returns:
        Renderer-specific OpenXR teleop session, or ``None`` when no active
        visualizer exposes ``configure_openxr_teleop``.
    """
    teleop_cfg = getattr(env_cfg, "isaac_teleop", None)
    if teleop_cfg is None:
        return None

    visualizers = getattr(getattr(env, "sim", None), "visualizers", [])
    for visualizer in visualizers:
        configure = getattr(visualizer, "configure_openxr_teleop", None)
        if not callable(configure):
            continue

        session = configure(env_cfg=env_cfg, env=env)
        if session is None:
            continue

        handles_provider = getattr(session, "openxr_handles_provider", None)
        if callable(handles_provider):
            teleop_cfg.openxr_handles_provider = handles_provider
            logger.info("Configured IsaacTeleop to use %s OpenXR handles.", type(visualizer).__name__)
        return session

    return None


def attach_renderer_openxr_anchor_provider(teleop_interface, renderer_openxr_session: object | None) -> None:
    """Attach a renderer-owned OpenXR anchor provider to an IsaacTeleop device.

    Args:
        teleop_interface: Teleop device that may accept an anchor provider.
        renderer_openxr_session: Renderer-specific OpenXR teleop session.
    """
    if renderer_openxr_session is None:
        return

    set_provider = getattr(teleop_interface, "set_anchor_world_matrix_provider", None)
    anchor_provider = getattr(renderer_openxr_session, "anchor_world_matrix", None)
    if callable(set_provider) and callable(anchor_provider):
        set_provider(anchor_provider)


class KitlessTeleopLauncher:
    """Coordinate kitless teleop lifecycle for scripts.

    This class provides the non-Kit equivalent of the small subset of
    AppLauncher behavior that teleop scripts need: process preflight, CloudXR
    startup, Kit launcher argument adjustment, and renderer-owned OpenXR handle
    wiring. Renderer-specific policy still lives in the visualizer runtime.
    """

    def __init__(
        self,
        launcher_args: argparse.Namespace,
        original_argv: list[str],
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the launcher and run any required process preflight.

        Args:
            launcher_args: Parsed script and AppLauncher arguments.
            original_argv: Process arguments before Hydra and preset folding.
            logger: Optional logger for diagnostics.
        """
        self.launcher_args = launcher_args
        self.original_argv = list(original_argv)
        self._logger = logger or logging.getLogger(__name__)
        self.runtime = renderer_openxr_runtime_for_args(launcher_args)
        reexec_with_packaged_openxr_loader_if_needed(
            enabled=self.runtime is not None and self.runtime.loader_required,
            argv=self.original_argv,
        )

    @property
    def enabled(self) -> bool:
        """Whether a renderer-owned OpenXR teleop runtime is active."""
        return self.runtime is not None

    @property
    def display_name(self) -> str:
        """Human-readable runtime name for diagnostics."""
        return self.runtime.display_name if self.runtime is not None else "renderer"

    def resolve_env_cfg(self, task: str, resolver: Callable[[str, str], object], current_argv: list[str]) -> object:
        """Resolve an environment config, stripping fallback presets if needed.

        Args:
            task: Task name to resolve.
            resolver: Callable matching ``resolve_task_config(task, entry_point_key)``.
            current_argv: Current process arguments after preset folding.

        Returns:
            The resolver return value.
        """
        try:
            return resolver(task, "")
        except ValueError as exc:
            if self.runtime is None or "Unknown preset(s)" not in str(exc):
                raise
            original_argv = list(current_argv)
            retry_argv = [
                current_argv[0],
                *strip_preset_tokens(current_argv[1:], self.runtime.fallback_presets),
            ]
            if retry_argv == original_argv:
                raise
            self._logger.info(
                "Task %s does not expose kitless %s preset names; retrying without fallback tokens.",
                task,
                self.runtime.display_name,
            )
            try:
                sys.argv = retry_argv
                return resolver(task, "")
            finally:
                sys.argv = original_argv

    def configure_env_cfg(self, env_cfg, current_argv: list[str]) -> bool:
        """Apply renderer-owned OpenXR defaults to an environment config.

        Args:
            env_cfg: Isaac Lab environment configuration.
            current_argv: Current process arguments after preset folding.

        Returns:
            Whether the renderer runtime physics backend is active.
        """
        if self.runtime is None:
            return False

        runtime_physics_enabled = self.runtime.physics_requested(
            self.launcher_args, self.original_argv, current_argv
        ) or self.runtime.env_cfg_uses_runtime_physics(env_cfg)
        self.runtime.apply_backend(env_cfg, enabled=runtime_physics_enabled)
        apply_renderer_owned_teleop_overrides(env_cfg, render_interval=self.runtime.render_interval)
        self.runtime.default_visualization_device(env_cfg, enabled=True)
        self.runtime.log_backend_selection(env_cfg, enabled=True)
        return runtime_physics_enabled

    def simulation_launcher_args(self) -> argparse.Namespace:
        """Return AppLauncher arguments for the active runtime."""
        return launcher_args_for_renderer_owned_openxr(self.launcher_args, runtime=self.runtime)

    def resolve_cloudxr_env(self, value: str | None) -> str | None:
        """Resolve a CloudXR profile path or shorthand."""
        return resolve_cloudxr_env(value)

    def launch_cloudxr(
        self,
        *,
        use_isaac_teleop: bool,
        cloudxr_env_path: str | None,
        auto_launch: bool,
    ) -> object | None:
        """Launch CloudXR when renderer-owned IsaacTeleop needs it.

        Args:
            use_isaac_teleop: Whether the IsaacTeleop stack is active.
            cloudxr_env_path: Resolved CloudXR profile path.
            auto_launch: Whether CloudXR auto-launch is enabled.

        Returns:
            The CloudXR launcher instance, or ``None``.
        """
        if not use_isaac_teleop or self.runtime is None:
            return None
        return launch_cloudxr_runtime(
            cloudxr_env_path,
            auto_launch=auto_launch,
            label=self.runtime.cloudxr_label,
        )

    def stop_cloudxr(self, cloudxr_launcher: object | None) -> None:
        """Stop a CloudXR launcher created by :meth:`launch_cloudxr`."""
        if cloudxr_launcher is None:
            return
        try:
            cloudxr_launcher.stop()
        except RuntimeError:
            self._logger.warning("CloudXR runtime process could not be terminated; handle retained for atexit cleanup")

    def configure_openxr_teleop(self, env, env_cfg, *, enabled: bool) -> object | None:
        """Wire renderer-owned OpenXR handles into IsaacTeleop.

        Args:
            env: Isaac Lab environment with initialized visualizers.
            env_cfg: Isaac Lab environment configuration.
            enabled: Whether OpenXR teleop wiring should be attempted.

        Returns:
            Renderer-specific OpenXR teleop session, or ``None``.
        """
        if not enabled or self.runtime is None:
            return None

        session = configure_renderer_openxr_teleop(env, env_cfg)
        teleop_cfg = getattr(env_cfg, "isaac_teleop", None)
        if teleop_cfg is not None and teleop_cfg.openxr_handles_provider is None:
            visualizer_names = ", ".join(sorted(visualizer_types_from_args(self.launcher_args))) or "<none>"
            raise RuntimeError(
                "Kitless XR requires an OpenXR-capable visualizer that exposes "
                f"configure_openxr_teleop(); requested visualizers: {visualizer_names}."
            )
        return session

    def attach_anchor_provider(self, teleop_interface, renderer_openxr_session: object | None) -> None:
        """Attach renderer-owned OpenXR anchor data to an IsaacTeleop device."""
        attach_renderer_openxr_anchor_provider(teleop_interface, renderer_openxr_session)

    def env_uses_runtime_physics(self, env) -> bool:
        """Return whether the active environment uses runtime-specific physics."""
        return self.runtime.env_uses_runtime_physics(env) if self.runtime is not None else False

    def debug_idle_for_env_cfg(self, env_cfg) -> bool:
        """Return whether synthetic idle teleop should be used before env creation."""
        return (
            self.runtime is not None
            and self.runtime.debug_idle_enabled()
            and self.runtime.env_cfg_uses_runtime_physics(env_cfg)
        )

    def debug_idle_for_env(self, env) -> bool:
        """Return whether synthetic idle teleop should be used for an environment."""
        return self.runtime is not None and self.runtime.debug_idle_enabled() and self.env_uses_runtime_physics(env)

    def debug_step_limit(self) -> int | None:
        """Return the synthetic idle-action debug step limit, if configured."""
        return self.runtime.debug_step_limit(self._logger) if self.runtime is not None else None

    def is_loop_running(self, env) -> bool:
        """Return whether a teleop loop should keep running."""
        visualizers = getattr(getattr(env, "sim", None), "visualizers", [])
        if visualizers:
            return any(viz.is_running() and not viz.is_closed for viz in visualizers)
        return True
