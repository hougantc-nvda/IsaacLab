# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils.configclass import configclass


@configclass
class NewtonIKManagerCfg:
    """Configuration for the Newton inverse-kinematics manager."""

    command_type: str = "pose"
    """IK command type. Supported values are ``"position"`` and ``"pose"``."""

    use_relative_mode: bool = True
    """Whether input commands are relative to the current end-effector pose."""

    optimizer: str = "lm"
    """Newton IK optimizer backend. Supported values are ``"lm"`` and ``"lbfgs"``."""

    jacobian_mode: str = "analytic"
    """Newton IK Jacobian backend. Supported values are ``"analytic"``, ``"autodiff"``, and ``"mixed"``."""

    iterations: int = 24
    """Number of Newton IK solver iterations per action application."""

    step_size: float = 1.0
    """Step size passed to Newton ``IKSolver.step``."""

    lambda_initial: float = 0.1
    """Initial damping value for the Newton Levenberg-Marquardt optimizer."""

    position_weight: float = 1.0
    """Residual weight for the end-effector position objective."""

    rotation_weight: float = 1.0
    """Residual weight for the end-effector rotation objective."""

    joint_limit_weight: float = 0.1
    """Residual weight for the joint-limit objective."""
