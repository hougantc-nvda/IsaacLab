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


def success(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate when the hose is inserted successfully."""
    return observations.insert_done(env)
