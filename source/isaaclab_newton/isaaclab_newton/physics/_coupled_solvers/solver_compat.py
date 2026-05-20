# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility metadata for Newton solvers used by the local coupled fallback."""

from __future__ import annotations


def apply_solver_coupling_compat(coupling_interface) -> None:
    """Apply PR-2848 coupling metadata to normal Newton solver classes."""
    try:
        from newton.solvers import SolverKamino, SolverMuJoCo  # noqa: PLC0415
    except ImportError:
        return

    unsupported_harvest = frozenset(
        {
            coupling_interface.Hook.BODY_PROXY_HARVEST,
            coupling_interface.Hook.PARTICLE_PROXY_HARVEST,
        }
    )
    for solver_class in (SolverMuJoCo, SolverKamino):
        if not hasattr(solver_class, "coupling_unsupported"):
            solver_class.coupling_unsupported = unsupported_harvest
