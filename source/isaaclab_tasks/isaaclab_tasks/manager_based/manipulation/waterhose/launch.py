# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch helpers for the Newton-based RBY1 waterhose task."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFER_NEWTON_IMPORT_ENV = "ISAACLAB_WATERHOSE_DEFER_NEWTON_IMPORT"


@dataclass(frozen=True)
class WaterhoseLaunchMode:
    """Resolved launch mode for scripts that can run the waterhose task."""

    uses_waterhose_task: bool
    visualizer_types: set[str]
    uses_kit_visualizer: bool
    uses_kitless_waterhose: bool


def get_visualizer_types(launcher_args: argparse.Namespace | dict | None) -> set[str]:
    """Return normalized visualizer names from launcher arguments.

    Args:
        launcher_args: Parsed launcher arguments.

    Returns:
        Lowercase visualizer names.
    """
    if isinstance(launcher_args, argparse.Namespace):
        visualizers = getattr(launcher_args, "visualizer", None)
    elif isinstance(launcher_args, dict):
        visualizers = launcher_args.get("visualizer")
    else:
        visualizers = None

    if not visualizers:
        return set()
    if isinstance(visualizers, str):
        visualizers = [token.strip() for token in visualizers.split(",")]
    return {str(visualizer).strip().lower() for visualizer in visualizers if str(visualizer).strip()}


def get_dataset_env_name(input_file: str | None) -> str | None:
    """Read the environment name from dataset metadata.

    Args:
        input_file: Dataset path.

    Returns:
        Environment name from the HDF5 ``env_args`` metadata, or ``None``.
    """
    if input_file is None or not Path(input_file).is_file():
        return None

    import h5py

    try:
        with h5py.File(input_file, "r") as dataset_file:
            env_args = dataset_file["data"].attrs.get("env_args")
    except (KeyError, OSError):
        return None

    if env_args is None:
        return None
    if isinstance(env_args, bytes):
        env_args = env_args.decode("utf-8")
    try:
        return json.loads(env_args).get("env_name")
    except (TypeError, json.JSONDecodeError):
        return None


def is_waterhose_task(task_name: str | None) -> bool:
    """Return whether a task name identifies the RBY1 waterhose task."""
    if task_name is None:
        return False
    return "waterhose" in task_name.split(":")[-1].lower()


def resolve_waterhose_task_name(task_name: str | None, input_file: str | None = None) -> str | None:
    """Resolve a waterhose-capable task name from CLI or dataset metadata.

    Args:
        task_name: CLI task name.
        input_file: Optional dataset path used when ``task_name`` is absent.

    Returns:
        The CLI task name when provided, otherwise the dataset environment name.
    """
    if task_name is not None:
        return task_name.split(":")[-1]
    return get_dataset_env_name(input_file)


def should_defer_newton_import() -> bool:
    """Return whether waterhose Newton/PXR imports should wait for Kit startup."""
    return os.environ.get(DEFER_NEWTON_IMPORT_ENV) == "1" or "kit" in get_cli_visualizer_types()


def get_cli_visualizer_types(argv: list[str] | None = None) -> set[str]:
    """Return visualizer types requested on the raw process command line."""
    argv = sys.argv[1:] if argv is None else argv
    visualizer_values: list[str] = []
    visualizer_flags = {"--visualizer", "--viz", "--vis"}
    for index, token in enumerate(argv):
        if token in visualizer_flags and index + 1 < len(argv):
            visualizer_values.append(argv[index + 1])
        elif any(token.startswith(f"{flag}=") for flag in visualizer_flags):
            visualizer_values.append(token.split("=", 1)[1])

    visualizers: set[str] = set()
    for value in visualizer_values:
        visualizers.update(part.strip().lower() for part in value.split(",") if part.strip())
    return visualizers


def import_waterhose_newton_dependencies() -> None:
    """Import Newton dependencies needed by the waterhose builder."""
    from . import waterhose_core as core

    core.import_newton_dependencies()


def prepare_waterhose_launch(
    launcher_args: argparse.Namespace | dict,
    *,
    task_name: str | None = None,
    input_file: str | None = None,
    parser: argparse.ArgumentParser | None = None,
    default_standalone_spacemouse: bool = False,
    require_standalone_spacemouse: bool = False,
    standalone_spacemouse_error: str | None = None,
) -> WaterhoseLaunchMode:
    """Prepare process-global launch state for waterhose scripts.

    The waterhose task has two valid import orders:

    * Kit visualizer path: launch Kit first, then import Newton/PXR-facing
      modules.
    * Kitless Newton path: import Newton before the broader Isaac Lab stack so
      the USD cable importer initializes consistently.

    Args:
        launcher_args: Parsed launcher arguments.
        task_name: Optional explicit task name. Defaults to
            ``launcher_args.task``.
        input_file: Optional dataset path used to infer the task.
        parser: Parser used for validation errors.
        default_standalone_spacemouse: Whether to set ``teleop_device`` to
            ``"spacemouse"`` for kitless waterhose runs when omitted.
        require_standalone_spacemouse: Whether kitless waterhose teleoperation
            must use a SpaceMouse.
        standalone_spacemouse_error: Validation message for non-SpaceMouse
            kitless waterhose teleoperation.

    Returns:
        Resolved launch mode.
    """
    visualizer_types = get_visualizer_types(launcher_args)
    if visualizer_types & {"kit", "newton"}:
        os.environ.setdefault("DISPLAY", ":1")

    if task_name is None:
        task_name = _get_arg(launcher_args, "task")
    resolved_task_name = resolve_waterhose_task_name(task_name, input_file)
    uses_waterhose_task = is_waterhose_task(resolved_task_name)
    uses_kit_visualizer = "kit" in visualizer_types
    uses_kitless_waterhose = uses_waterhose_task and not uses_kit_visualizer

    if uses_waterhose_task and uses_kit_visualizer:
        os.environ[DEFER_NEWTON_IMPORT_ENV] = "1"

    if uses_kitless_waterhose:
        if default_standalone_spacemouse and _get_arg(launcher_args, "teleop_device") is None:
            _set_arg(launcher_args, "teleop_device", "spacemouse")
        if require_standalone_spacemouse:
            teleop_device = _get_arg(launcher_args, "teleop_device")
            if teleop_device is not None and str(teleop_device).lower() != "spacemouse":
                message = standalone_spacemouse_error or (
                    "Waterhose teleoperation with the Newton viewer requires --teleop_device spacemouse."
                )
                if parser is not None:
                    parser.error(message)
                raise ValueError(message)
        import_waterhose_newton_dependencies()

    return WaterhoseLaunchMode(
        uses_waterhose_task=uses_waterhose_task,
        visualizer_types=visualizer_types,
        uses_kit_visualizer=uses_kit_visualizer,
        uses_kitless_waterhose=uses_kitless_waterhose,
    )


def _get_arg(launcher_args: argparse.Namespace | dict, name: str) -> Any:
    if isinstance(launcher_args, argparse.Namespace):
        return getattr(launcher_args, name, None)
    return launcher_args.get(name)


def _set_arg(launcher_args: argparse.Namespace | dict, name: str, value: Any) -> None:
    if isinstance(launcher_args, argparse.Namespace):
        setattr(launcher_args, name, value)
    else:
        launcher_args[name] = value
