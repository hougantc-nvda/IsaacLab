# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import observations

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reach_hose(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward the TCP approaching the hose plug."""
    dist = torch.linalg.norm(observations.eef_pos(env) - observations.plug_pos(env), dim=-1)
    return torch.exp(-20.0 * dist)


def align_tip(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward low lateral alignment error and axis alignment."""
    errors = observations.alignment(env)
    return torch.exp(-600.0 * errors[:, 0]) + 0.5 * (1.0 - errors[:, 2]).clamp(min=0.0)


def insert_tip(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward insertion progress into the socket."""
    return observations.alignment(env)[:, 1].clamp(min=0.0)


def success_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward successful insertion."""
    return observations.insert_done(env).float()


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize changes in task-space actions."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
