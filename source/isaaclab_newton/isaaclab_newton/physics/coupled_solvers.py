# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resolve Newton coupled solver classes with a local fallback.

Newton PR 2848 introduced ``newton.solvers.coupled_experimental``. Isaac Lab
uses the upstream implementation when available and falls back to the local copy
while normal Newton releases do not yet ship that module.
"""

from __future__ import annotations

try:
    from newton.solvers.coupled_experimental import (  # type: ignore[import-not-found]
        CouplingInterface,
        ModelView,
        SolverAdmmCoupled,
        SolverCoupled,
        SolverProxyCoupled,
    )

    USING_UPSTREAM_COUPLED_SOLVERS = True
except ImportError:
    from ._coupled_solvers import (
        CouplingInterface,
        ModelView,
        SolverAdmmCoupled,
        SolverCoupled,
        SolverProxyCoupled,
    )
    from ._coupled_solvers.mpm_compat import apply_mpm_proxy_compat
    from ._coupled_solvers.solver_compat import apply_solver_coupling_compat

    apply_solver_coupling_compat(CouplingInterface)
    apply_mpm_proxy_compat()
    USING_UPSTREAM_COUPLED_SOLVERS = False


__all__ = [
    "CouplingInterface",
    "ModelView",
    "SolverAdmmCoupled",
    "SolverCoupled",
    "SolverProxyCoupled",
    "USING_UPSTREAM_COUPLED_SOLVERS",
]
