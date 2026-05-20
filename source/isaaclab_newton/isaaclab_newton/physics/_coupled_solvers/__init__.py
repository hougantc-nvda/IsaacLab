# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vendored fallback exports for Newton coupled solvers."""

from .interface import CouplingInterface
from .model_view import ModelView
from .solver_admm_coupled import SolverAdmmCoupled
from .solver_coupled import SolverCoupled
from .solver_proxy_coupled import SolverProxyCoupled

__all__ = [
    "CouplingInterface",
    "ModelView",
    "SolverAdmmCoupled",
    "SolverCoupled",
    "SolverProxyCoupled",
]
