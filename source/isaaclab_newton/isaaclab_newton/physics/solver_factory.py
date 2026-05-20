# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for constructing Newton solvers from Isaac Lab solver configs."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable

from newton.solvers import SolverFeatherstone, SolverImplicitMPM, SolverKamino, SolverMuJoCo, SolverXPBD

from .mjwarp_manager import resolve_mujoco_solver_kwargs
from .newton_manager_cfg import NewtonSolverCfg

_SOLVER_CLASS_BY_TYPE = {
    "mujoco_warp": SolverMuJoCo,
    "implicit_mpm": SolverImplicitMPM,
    "xpbd": SolverXPBD,
    "featherstone": SolverFeatherstone,
    "kamino": SolverKamino,
}


def resolve_class_or_callable(value: type | Callable | str) -> type | Callable:
    """Resolve a callable, LazyType-like object, or ``module:attr`` string."""
    if isinstance(value, str):
        module_name, _, attr = value.partition(":")
        if not module_name or not attr:
            raise ValueError(f"Expected callable as 'module:attr', got {value!r}.")
        return getattr(importlib.import_module(module_name), attr)
    if hasattr(value, "_resolve"):
        return value._resolve()
    return value


def resolve_newton_solver_class(solver_cfg: NewtonSolverCfg, solver_class: type | Callable | str | None = None):
    """Resolve the Newton solver class for a solver config."""
    if solver_class is not None:
        return resolve_class_or_callable(solver_class)

    solver_type = getattr(solver_cfg, "solver_type", None)
    resolved = _SOLVER_CLASS_BY_TYPE.get(solver_type)
    if resolved is None:
        raise ValueError(
            f"Cannot infer Newton solver class from solver_type={solver_type!r}. "
            "Set CoupledSolverEntryCfg.solver_class for custom solvers."
        )
    return resolved


def resolve_newton_solver_kwargs(solver_cfg: NewtonSolverCfg, solver_class: type | Callable | None = None) -> dict:
    """Translate an Isaac Lab Newton solver config into constructor kwargs."""
    solver_type = getattr(solver_cfg, "solver_type", None)
    if solver_type == "implicit_mpm":
        return {"config": solver_cfg.to_solver_config()}
    if solver_type == "mujoco_warp":
        return resolve_mujoco_solver_kwargs(solver_cfg)

    if solver_class is None:
        solver_class = resolve_newton_solver_class(solver_cfg)
    signature_target = solver_class.__init__ if inspect.isclass(solver_class) else solver_class
    valid = set(inspect.signature(signature_target).parameters) - {"self", "model"}
    return {key: value for key, value in solver_cfg.to_dict().items() if key in valid}


def resolve_newton_solver_class_and_kwargs(
    solver_cfg: NewtonSolverCfg,
    solver_class: type | Callable | str | None = None,
    solver_kwargs: dict | None = None,
):
    """Resolve solver class and kwargs, with explicit kwargs taking precedence."""
    resolved_class = resolve_newton_solver_class(solver_cfg, solver_class)
    resolved_kwargs = resolve_newton_solver_kwargs(solver_cfg, resolved_class)
    if solver_kwargs:
        resolved_kwargs.update(solver_kwargs)
    return resolved_class, resolved_kwargs


def solver_cfg_needs_external_contacts(solver_cfg: NewtonSolverCfg) -> bool:
    """Infer whether a solver config expects contacts from Newton's outer collision pipeline."""
    solver_type = getattr(solver_cfg, "solver_type", None)
    if solver_type == "mujoco_warp":
        return not getattr(solver_cfg, "use_mujoco_contacts", True)
    if solver_type == "kamino":
        return not getattr(solver_cfg, "use_collision_detector", True)
    return solver_type in ("xpbd", "featherstone")
