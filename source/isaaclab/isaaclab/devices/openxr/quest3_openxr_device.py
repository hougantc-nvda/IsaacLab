# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenXR-powered device for teleoperation and interaction."""

import contextlib
import numpy as np
from dataclasses import dataclass
from typing import Any

import carb
import usdrt
from pxr import Gf as pxrGf
from usdrt import Rt

from isaaclab.devices.retargeter_base import RetargeterBase

from .openxr_device import OpenXRDevice, OpenXRDeviceCfg

#with contextlib.suppress(ModuleNotFoundError):
from omni.kit.xr.core import XRCore, XRInputDevice

# Extend TrackingTarget enum for controllers
from enum import Enum

# Create a new enum that includes all TrackingTarget values plus new ones
class Quest3TrackingTarget(Enum):
    """Extended tracking targets for Quest3 controllers."""
    CONTROLLER_LEFT = len(OpenXRDevice.TrackingTarget)
    CONTROLLER_RIGHT = len(OpenXRDevice.TrackingTarget) + 1


@dataclass
class Quest3OpenXRDeviceCfg(OpenXRDeviceCfg):
    """Configuration for Quest3 OpenXR devices."""

    pass


class Quest3OpenXRDevice(OpenXRDevice):

    def __init__(
        self,
        cfg: Quest3OpenXRDeviceCfg,
        retargeters: list[RetargeterBase] | None = None,
    ):
        """Initialize the OpenXR device.

        Args:
            cfg: Configuration object for OpenXR settings.
            retargeters: List of retargeter instances to use for transforming raw tracking data.
        """
        super().__init__(cfg, retargeters)

    """
    Operations
    """

    def reset(self):
        super().reset()

    def _get_raw_data(self) -> Any:
        """Get the latest tracking data from the OpenXR runtime.

        Returns:
            Dictionary with TrackingTarget enum keys (HAND_LEFT, HAND_RIGHT, HEAD) containing:
                - Left motion controller pose: Dictionary of 26 joints with position and orientation
                - Right motion controller pose: Dictionary of 26 joints with position and orientation
                - Head pose: Single 7-element array with position and orientation

        Each pose is represented as a 7-element array: [x, y, z, qw, qx, qy, qz]
        where the first 3 elements are position and the last 4 are quaternion orientation.
        """
        data = super()._get_raw_data()

        left_input_device = XRCore.get_singleton().get_input_device("/user/hand/left")
        right_input_device = XRCore.get_singleton().get_input_device("/user/hand/right")

        data.update({
            Quest3TrackingTarget.CONTROLLER_LEFT: self._query_controller_input_values(
                Quest3TrackingTarget.CONTROLLER_LEFT,
                left_input_device
            ),
            Quest3TrackingTarget.CONTROLLER_RIGHT: self._query_controller_input_values(
                Quest3TrackingTarget.CONTROLLER_RIGHT,
                right_input_device
            )
        })

        # Since we are using controller data, we need to overwrite the wrist pose.
        self._set_wrist_pose(left_input_device, data[OpenXRDevice.TrackingTarget.HAND_LEFT])
        self._set_wrist_pose(right_input_device, data[OpenXRDevice.TrackingTarget.HAND_RIGHT])
        return data

    """
    Internal helpers.
    """

    def _query_controller_input_values(
        self, tracking_target : Quest3TrackingTarget, input_device
    ) -> np.array:
        """Calculate and update input device data

        """

        if input_device is None:
            return np.array([])

        thumbstick_x = 0.0
        thumbstick_y = 0.0
        trigger = 0.0
        squeeze = 0.0
        a = 0.0
        b = 0.0
        x = 0.0
        y = 0.0

        if input_device.has_input_gesture("thumbstick", "x"):
            thumbstick_x: float = input_device.get_input_gesture_value("thumbstick", "x")

        if input_device.has_input_gesture("thumbstick", "y"):
            thumbstick_y: float = input_device.get_input_gesture_value("thumbstick", "y")

        if input_device.has_input_gesture("trigger", "value"):
            trigger: float = input_device.get_input_gesture_value("trigger", "value")

        if input_device.has_input_gesture("squeeze", "value"):
            squeeze: float = input_device.get_input_gesture_value("squeeze", "value")

        if input_device.has_input_gesture("a", "click"):
            a: float = input_device.get_input_gesture_value("a", "click")

        if input_device.has_input_gesture("b", "click"):
            b: float = input_device.get_input_gesture_value("b", "click")

        if input_device.has_input_gesture("x", "click"):
            x: float = input_device.get_input_gesture_value("x", "click")

        if input_device.has_input_gesture("y", "click"):
            y: float = input_device.get_input_gesture_value("y", "click")

        return np.array([thumbstick_x, thumbstick_y, trigger, squeeze, a, b, x, y], dtype=np.float32)

    def _set_wrist_pose(self, input_device, data):
        """Set the wrist pose in the data dictionary."""
        if input_device is not None and data is not None and "wrist" in data:
            pose = input_device.get_virtual_world_pose()
            position = pose.ExtractTranslation()
            quat = pose.ExtractRotationQuat()
            data["wrist"] = np.array([position[0], position[1], position[2], quat.GetReal(), quat.GetImaginary()[0], quat.GetImaginary()[1], quat.GetImaginary()[2]], dtype=np.float32)