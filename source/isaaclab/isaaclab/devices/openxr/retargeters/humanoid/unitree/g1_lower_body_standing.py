# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import dataclass

from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab.devices.openxr.quest3_openxr_device import Quest3TrackingTarget
from isaaclab.sim import SimulationContext


@dataclass
class G1LowerBodyStandingRetargeterCfg(RetargeterCfg):
    """Configuration for the G1 lower body standing retargeter."""

    hip_height: float = 0.72
    """Height of the G1 robot hip in meters. The value is a fixed height suitable for G1 to do tabletop manipulation."""


class G1LowerBodyStandingRetargeter(RetargeterBase):
    """Provides lower body standing commands for the G1 robot."""

    def __init__(self, cfg: G1LowerBodyStandingRetargeterCfg):
        """Initialize the retargeter."""
        self.cfg = cfg
        self._hip_height = cfg.hip_height

    def retarget(self, data: dict) -> torch.Tensor:
        left_thumbstick_x = 0.0
        left_thumbstick_y = 0.0
        right_thumbstick_x = 0.0
        right_thumbstick_y = 0.0

        # TODO: Make the indices defined in the enum. Currently these are coming from quest3_openxr_device.py.
        if Quest3TrackingTarget.CONTROLLER_LEFT in data and data[Quest3TrackingTarget.CONTROLLER_LEFT] is not None:
            left_hand_data = data[Quest3TrackingTarget.CONTROLLER_LEFT]            
            if len(left_hand_data) >= 2:
                left_thumbstick_x = left_hand_data[0]
                left_thumbstick_y = left_hand_data[1]

        if Quest3TrackingTarget.CONTROLLER_RIGHT in data and data[Quest3TrackingTarget.CONTROLLER_RIGHT] is not None:
            right_hand_data = data[Quest3TrackingTarget.CONTROLLER_RIGHT]
            if len(right_hand_data) >= 2:
                right_thumbstick_x = right_hand_data[0]
                right_thumbstick_y = right_hand_data[1]

        dt = SimulationContext.instance().get_physics_dt()
        self._hip_height -= right_thumbstick_y * dt

        # print(f"left_thumbstick_x: {left_thumbstick_x}, left_thumbstick_y: {left_thumbstick_y}, right_thumbstick_x: {right_thumbstick_x}, right_thumbstick_y: {right_thumbstick_y}, self._hip_height: {self._hip_height}")

        return torch.tensor([-left_thumbstick_y, -left_thumbstick_x, -right_thumbstick_x, self._hip_height], device=self.cfg.sim_device, dtype=torch.float32)
