# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the GR1T2 pick-place task configuration."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from isaaclab_ovphysx.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab_tasks.contrib.pick_place.pickplace_gr1t2_env_cfg import PickPlaceGR1T2EnvCfg, PickPlaceGR1T2PhysicsCfg
from isaaclab_tasks.utils import resolve_task_config

_EXPECTED_HAND_JOINT_NAMES = [
    "L_index_proximal_joint",
    "L_index_intermediate_joint",
    "L_middle_proximal_joint",
    "L_middle_intermediate_joint",
    "L_pinky_proximal_joint",
    "L_pinky_intermediate_joint",
    "L_ring_proximal_joint",
    "L_ring_intermediate_joint",
    "L_thumb_proximal_yaw_joint",
    "L_thumb_proximal_pitch_joint",
    "L_thumb_distal_joint",
    "R_index_proximal_joint",
    "R_index_intermediate_joint",
    "R_middle_proximal_joint",
    "R_middle_intermediate_joint",
    "R_pinky_proximal_joint",
    "R_pinky_intermediate_joint",
    "R_ring_proximal_joint",
    "R_ring_intermediate_joint",
    "R_thumb_proximal_yaw_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_distal_joint",
]


def test_gr1t2_pickplace_exposes_ovphysx_physics_preset():
    """Verify the GR1T2 task can resolve the kitless OvPhysX backend."""
    physics_cfg = PickPlaceGR1T2PhysicsCfg()
    assert isinstance(physics_cfg.default, PhysxCfg)
    assert isinstance(physics_cfg.physx, PhysxCfg)
    assert isinstance(physics_cfg.ovphysx, OvPhysxCfg)

    original_argv = sys.argv.copy()
    try:
        sys.argv = [sys.argv[0], "presets=ovphysx"]
        cfg, _ = resolve_task_config("Isaac-PickPlace-GR1T2-Abs-v0", "")
    finally:
        sys.argv = original_argv

    assert isinstance(cfg.sim.physics, OvPhysxCfg)


def test_gr1t2_pickplace_kitless_teleop_uses_packaged_kinematics_urdf():
    """Verify kitless teleop can configure Pink IK without a Kit USD-to-URDF conversion."""
    cfg = PickPlaceGR1T2EnvCfg()
    cfg.apply_kitless_teleop_overrides()

    controller = cfg.actions.upper_body_ik.controller
    assert controller.urdf_path is not None
    assert Path(controller.urdf_path).is_file()
    assert Path(controller.urdf_path).name == "GR1T2_fourier_hand_6dof_kinematics.urdf"
    assert controller.mesh_path is None
    assert controller.usd_path is None
    assert cfg.scene.object.prim_path == "{ENV_REGEX_NS}/Object"
    assert cfg.scene.object.spawn.usd_path.endswith("steering_wheel.usd")
    assert cfg.scene.packing_table.spawn.usd_path.endswith("packing_table.usd")


def test_gr1t2_pickplace_robot_actuates_head_and_legs():
    """Verify the task does not leave head and leg joints unactuated."""
    cfg = PickPlaceGR1T2EnvCfg()

    assert "head" in cfg.scene.robot.actuators
    assert "legs" in cfg.scene.robot.actuators
    assert cfg.scene.robot.actuators["head"].joint_names_expr == ["head_.*"]
    assert cfg.scene.robot.actuators["legs"].joint_names_expr == [".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"]
    assert cfg.scene.robot.actuators["head"].stiffness == 4400.0
    assert cfg.scene.robot.actuators["head"].damping == 40.0
    assert cfg.scene.robot.actuators["legs"].stiffness == 4400.0
    assert cfg.scene.robot.actuators["legs"].damping == 40.0


def test_gr1t2_pickplace_teleop_hand_order_matches_action_space():
    """Verify IsaacTeleop hand actions are flattened in Pink hand-joint order."""
    cfg = PickPlaceGR1T2EnvCfg()

    assert cfg.actions.upper_body_ik.hand_joint_names == _EXPECTED_HAND_JOINT_NAMES


def test_gr1t2_pickplace_kitless_newton_overrides_apply():
    """Verify GR1T2-owned Newton overrides are applied from the task config."""
    from isaaclab_newton.physics import NewtonManager

    try:
        NewtonManager.clear_usd_import_options()
        cfg = PickPlaceGR1T2EnvCfg()
        cfg.apply_kitless_newton_overrides()

        assert NewtonManager._usd_import_options["PackingTable"] == {
            "load_static_visual_shapes": True,
            "load_xform_collision_shapes": True,
            "floating": False,
        }
        assert NewtonManager._usd_import_options["Object"] == {
            "load_collision_visual_shapes": True,
            "hide_collision_shapes": True,
        }
        controller = cfg.actions.upper_body_ik.controller
        assert controller.urdf_path is not None
        assert Path(controller.urdf_path).is_file()
        assert controller.usd_path is None
        assert cfg.scene.robot.actuators["trunk"].stiffness == 500.0
        assert cfg.scene.robot.actuators["right-arm"].effort_limit_sim == 800.0
        assert controller.variable_input_tasks[0].lm_damping == 12.0
        assert controller.variable_input_tasks[0].gain == 0.5
    finally:
        NewtonManager.clear_usd_import_options()


def test_gr1t2_pickplace_flags_invalid_kitless_teleop_targets():
    """Verify task-owned action guards catch invalid wrist and hand targets."""
    cfg = PickPlaceGR1T2EnvCfg()
    identity_wrist_action = torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, *([0.0] * 22)],
        dtype=torch.float32,
    )
    tracked_wrist_action = identity_wrist_action.clone()
    tracked_wrist_action[0] = -0.2
    tracked_wrist_action[7] = 0.2
    bad_hand_action = tracked_wrist_action.clone()
    bad_hand_action[14] = 7.0

    assert "identity wrist targets" in cfg.teleop_action_target_issue(identity_wrist_action)
    assert "unreasonable hand joint target magnitude" in cfg.teleop_action_target_issue(bad_hand_action)
    assert cfg.teleop_action_target_issue(tracked_wrist_action) is None


def test_gr1t2_pickplace_conditions_kitless_wrist_targets_from_current_pose():
    """Verify kitless Newton wrist targets are ramped instead of applied as a jump."""
    cfg = PickPlaceGR1T2EnvCfg()
    body_names = ["base_link", "left_hand_pitch_link", "right_hand_pitch_link"]
    body_link_pose_w = torch.tensor(
        [
            [
                [0.0, 0.0, 0.93, 0.0, 0.0, 0.0, 1.0],
                [-0.25, 0.10, 1.20, 0.0, 0.0, 0.0, 1.0],
                [0.25, 0.10, 1.20, 0.0, 0.0, 0.0, 1.0],
            ]
        ],
        dtype=torch.float32,
    )
    robot = SimpleNamespace(body_names=body_names, data=SimpleNamespace(body_link_pose_w=body_link_pose_w))
    env = SimpleNamespace(
        scene={"robot": robot},
        cfg=SimpleNamespace(actions=SimpleNamespace(upper_body_ik=cfg.actions.upper_body_ik)),
    )
    action = torch.zeros((1, 36), dtype=torch.float32)
    action[0, 0:7] = torch.tensor([-0.40, 0.10, 1.50, 0.0, 0.0, 0.0, 1.0])
    action[0, 7:14] = torch.tensor([0.40, 0.10, 1.50, 0.0, 0.0, 0.0, 1.0])
    state: dict[str, torch.Tensor] = {}

    conditioned = cfg.condition_kitless_teleop_action(env, action, state)

    left_step = torch.linalg.norm(conditioned[0, 0:3] - body_link_pose_w[0, 1, :3])
    right_step = torch.linalg.norm(conditioned[0, 7:10] - body_link_pose_w[0, 2, :3])
    assert 0.0 < float(left_step) <= 0.0101
    assert 0.0 < float(right_step) <= 0.0101
    assert not torch.equal(conditioned[0, 0:3], action[0, 0:3])
    torch.testing.assert_close(conditioned[0, 14:], action[0, 14:])
