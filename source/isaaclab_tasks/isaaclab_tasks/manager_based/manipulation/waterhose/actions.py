# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import torch

from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor
from isaaclab.managers import ActionTermCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.utils.configclass import configclass

from isaaclab_newton.ik.newton_ik_manager import NewtonIKManager
from isaaclab_newton.ik.newton_ik_manager_cfg import NewtonIKManagerCfg

from . import waterhose_core as core


@configclass
class NewtonTaskSpaceIKActionCfg(ActionTermCfg):
    """Task-space RBY1 action backed by the Newton IK manager."""

    class_type: type[ActionTerm] = MISSING
    asset_name: str = "newton_waterhose"
    command_frame: str = "world"
    rotation_frame: str | None = None
    accumulate_targets: bool = False
    position_scale: float = 0.04
    rotation_scale: float = 0.25
    max_target_step: float = 0.018
    max_joint_step: float = 0.02
    max_gripper_joint_step: float = 0.20
    max_gripper_joint_velocity: float = 0.03
    ik_iterations: int = 4
    controller: NewtonIKManagerCfg = NewtonIKManagerCfg(
        command_type="pose",
        use_relative_mode=True,
        optimizer="lm",
        jacobian_mode="analytic",
        iterations=4,
        position_weight=1.0,
        rotation_weight=1.0,
        joint_limit_weight=10.0,
    )


class NewtonTaskSpaceIKAction(ActionTerm):
    """Apply relative end-effector pose and gripper commands to the Newton RBY1 robot."""

    cfg: NewtonTaskSpaceIKActionCfg

    def __init__(self, cfg: NewtonTaskSpaceIKActionCfg, env):
        # This task has no Isaac Lab Articulation asset; Newton owns the robot buffers.
        ManagerTermBase.__init__(self, cfg, env)
        self._IO_descriptor = GenericActionIODescriptor()
        self._export_IO_descriptor = True
        self._debug_vis_handle = None
        self._raw_actions = torch.zeros(self.num_envs, 7, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._scene_builder = env.waterhose_scene_builder
        self._model = core.NewtonManager.get_model()
        self._control = core.NewtonManager.get_control()
        self._single_robot_model = self._scene_builder.single_robot_model
        if self.cfg.command_frame not in ("eef", "world"):
            raise ValueError("NewtonTaskSpaceIKActionCfg.command_frame must be 'eef' or 'world'.")
        self._rotation_frame = (
            self.cfg.rotation_frame if self.cfg.rotation_frame is not None else self.cfg.command_frame
        )
        if self._rotation_frame not in ("eef", "world"):
            raise ValueError("NewtonTaskSpaceIKActionCfg.rotation_frame must be 'eef', 'world', or None.")
        self._body_ids = _resolve_body_ids(self._model.body_label, core.RIGHT_EE, self.num_envs)
        self._right_open_driver, self._right_closed_driver = _gripper_driver_targets(
            self._model, self._scene_builder.right_gripper_driver_dofs
        )
        self._left_open_driver, _ = _gripper_driver_targets(self._model, self._scene_builder.left_gripper_driver_dofs)
        self._init_tensor_views()
        self._init_static_tensors()
        self._setup_ik()
        self._seed_task_targets()
        self._seed_control_targets()

    @property
    def action_dim(self) -> int:
        return 7

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions.clamp(-1.0, 1.0)
        self._processed_actions[:, :3] = self._raw_actions[:, :3] * float(self.cfg.position_scale)
        self._processed_actions[:, 3:6] = self._raw_actions[:, 3:6] * float(self.cfg.rotation_scale)
        self._processed_actions[:, 6:] = self._raw_actions[:, 6:]

    def apply_actions(self) -> None:
        if self._env.sim.visualizers:
            core.apply_viewer_forces(self._env.sim)

        torch.index_select(self._body_q_t, 0, self._body_ids_t, out=self._ee_body_q)
        torch.take(self._joint_q_t, self._joint_coord_ids_t, out=self._current_q)

        action_pose = self._processed_actions[:, :6]
        if self.cfg.accumulate_targets:
            delta_pos, target_quat = self._target_delta_t(self._target_quat, action_pose)
            self._target_pos.add_(delta_pos)
            self._target_quat.copy_(target_quat)
            target_pos = self._target_pos
        else:
            delta_pos, target_quat = self._target_delta_t(self._ee_body_q[:, 3:7], action_pose)
            target_pos = self._ee_body_q[:, :3] + delta_pos

        self._ik_manager.set_target_pose(target_pos - self._env_origins_t, target_quat)
        solved_q = self._ik_manager.solve(self._current_q)

        self._next_control_q.copy_(self._nominal_control_q)
        self._next_control_q[:, self._ik_control_coord_ids_t] = solved_q[:, self._ik_control_coord_ids_t]
        self._set_gripper_targets_t(self._next_control_q, self._processed_actions[:, 6])
        self._joint_delta.copy_(self._next_control_q)
        self._joint_delta.sub_(self._last_control_q)
        self._joint_delta.clamp_(min=-self._max_step_t, max=self._max_step_t)
        self._last_control_q.add_(self._joint_delta)
        self._control_joint_target_pos_t.scatter_(0, self._joint_coord_ids_flat_t, self._last_control_q.reshape(-1))

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._reset_task_targets(env_ids)
        self._reset_control_targets(env_ids)

    def _init_tensor_views(self) -> None:
        state = core.NewtonManager.get_state_0()
        self._body_q_t = core.wp.to_torch(state.body_q)
        self._joint_q_t = core.wp.to_torch(state.joint_q)
        self._control_joint_target_pos_t = core.wp.to_torch(self._control.joint_target_pos)

    def _init_static_tensors(self) -> None:
        self._num_robot_coords = int(self._single_robot_model.joint_coord_count)
        self._body_ids_t = torch.as_tensor(self._body_ids, device=self.device, dtype=torch.long)
        self._all_env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._joint_coord_ids_t = torch.as_tensor(
            self._scene_builder.robot_joint_coord_ids_by_env,
            device=self.device,
            dtype=torch.long,
        )
        self._joint_coord_ids_flat_t = self._joint_coord_ids_t.reshape(-1)
        self._env_origins_t = torch.as_tensor(
            [[float(origin[i]) for i in range(3)] for origin in self._scene_builder.env_origins],
            device=self.device,
            dtype=torch.float32,
        )
        self._ee_body_q = torch.empty((self.num_envs, 7), device=self.device, dtype=torch.float32)
        self._current_q = torch.empty((self.num_envs, self._num_robot_coords), device=self.device, dtype=torch.float32)
        self._next_control_q = torch.empty_like(self._current_q)
        self._joint_delta = torch.empty_like(self._current_q)
        self._max_step_t = torch.full_like(self._current_q, float(self.cfg.max_joint_step))
        if self._scene_builder.gripper_dofs:
            self._max_step_t[:, self._scene_builder.gripper_dofs] = self._gripper_joint_step_limit()

        self._right_gripper_driver_dof = _first_index(self._scene_builder.right_gripper_driver_dofs)
        self._right_gripper_dofs = tuple(self._scene_builder.right_gripper_dofs[:2])
        self._left_gripper_driver_dof = _first_index(self._scene_builder.left_gripper_driver_dofs)
        self._left_gripper_dofs = tuple(self._scene_builder.left_gripper_dofs[:2])

    def target_pose_to_action(
        self, target_pos: torch.Tensor, target_quat_xyzw: torch.Tensor, env_id: int = 0
    ) -> torch.Tensor:
        curr = self._body_q_t[self._body_ids_t[env_id]]
        curr_pos = curr[:3]
        delta_pos = target_pos.to(device=self.device, dtype=torch.float32) - curr_pos
        curr_quat = curr[3:7]
        target_quat = target_quat_xyzw.to(device=self.device, dtype=torch.float32)
        if self.cfg.command_frame == "eef":
            delta_pos = _torch_quat_rotate(_torch_quat_inverse(curr_quat), delta_pos)
        if self._rotation_frame == "eef":
            delta_axis = _torch_axis_angle_between_eef(curr_quat, target_quat)
        else:
            delta_axis = _torch_axis_angle_between_world(curr_quat, target_quat)
        action = torch.zeros(7, device=self.device)
        action[:3] = delta_pos / float(self.cfg.position_scale)
        action[3:6] = delta_axis / float(self.cfg.rotation_scale)
        return action.clamp(-1.0, 1.0)

    def action_to_target_pose(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actions = actions.to(device=self.device, dtype=torch.float32)
        torch.index_select(
            self._body_q_t,
            0,
            self._body_ids_t[: actions.shape[0]],
            out=self._ee_body_q[: actions.shape[0]],
        )
        processed = torch.cat(
            (
                actions[:, :3] * float(self.cfg.position_scale),
                actions[:, 3:6] * float(self.cfg.rotation_scale),
            ),
            dim=-1,
        )
        delta_pos, quat = self._target_delta_t(self._ee_body_q[: actions.shape[0], 3:7], processed)
        return self._ee_body_q[: actions.shape[0], :3] + delta_pos, quat

    def _target_delta_t(self, ee_quat: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta_pos = action[:, :3] if action.ndim == 2 else action[:3]
        axis_angle = action[:, 3:6] if action.ndim == 2 else action[3:6]
        if self.cfg.command_frame == "eef":
            delta_pos = _torch_quat_rotate(ee_quat, delta_pos)
        if self._rotation_frame == "eef":
            target_quat = _torch_integrate_axis_angle_eef(ee_quat, axis_angle)
        else:
            target_quat = _torch_integrate_axis_angle_world(ee_quat, axis_angle)
        return delta_pos, target_quat

    def _seed_task_targets(self) -> None:
        torch.index_select(self._body_q_t, 0, self._body_ids_t, out=self._ee_body_q)
        self._target_pos = self._ee_body_q[:, :3].clone()
        self._target_quat = self._ee_body_q[:, 3:7].clone()

    def _reset_task_targets(self, env_ids) -> None:
        ids = self._env_indices_t(env_ids)
        if ids.numel() == self.num_envs:
            torch.index_select(self._body_q_t, 0, self._body_ids_t, out=self._ee_body_q)
            self._target_pos.copy_(self._ee_body_q[:, :3])
            self._target_quat.copy_(self._ee_body_q[:, 3:7])
            return
        body_ids = self._body_ids_t.index_select(0, ids)
        body_q = self._body_q_t.index_select(0, body_ids)
        self._target_pos[ids] = body_q[:, :3]
        self._target_quat[ids] = body_q[:, 3:7]

    def _setup_ik(self) -> None:
        link_index = _resolve_body_ids(self._single_robot_model.body_label, core.RIGHT_EE, 1)[0]
        self.cfg.controller.iterations = int(self.cfg.ik_iterations)
        self._ik_control_coord_ids = _right_arm_ik_coord_ids(self._single_robot_model)
        self._ik_control_coord_ids_t = torch.as_tensor(self._ik_control_coord_ids, device=self.device, dtype=torch.long)
        self._ik_manager = NewtonIKManager(
            self.cfg.controller,
            model=self._single_robot_model,
            num_envs=self.num_envs,
            device=self.device,
            link_index=link_index,
            link_offset_pos=(0.0, 0.0, 0.0),
            link_offset_rot=(0.0, 0.0, 0.0, 1.0),
        )

    def _seed_control_targets(self) -> None:
        initial = core.wp.to_torch(self._single_robot_model.joint_q).to(device=self.device, dtype=torch.float32)
        self._nominal_control_q = initial.unsqueeze(0).repeat(self.num_envs, 1).contiguous()
        self._last_control_q = self._nominal_control_q.clone()
        self._control_joint_target_pos_t.scatter_(0, self._joint_coord_ids_flat_t, self._last_control_q.reshape(-1))

    def _set_gripper_targets_t(self, q: torch.Tensor, gripper_action: torch.Tensor) -> None:
        right_alpha = (gripper_action > 0.0).to(dtype=torch.float32)
        right_driver = (1.0 - right_alpha) * self._right_open_driver + right_alpha * self._right_closed_driver
        if self._right_gripper_driver_dof is not None:
            q[:, self._right_gripper_driver_dof] = right_driver
        if len(self._right_gripper_dofs) >= 2:
            q[:, self._right_gripper_dofs[0]] = -0.5 * right_driver
            q[:, self._right_gripper_dofs[1]] = 0.5 * right_driver
        if self._left_gripper_driver_dof is not None:
            q[:, self._left_gripper_driver_dof] = self._left_open_driver
        if len(self._left_gripper_dofs) >= 2:
            q[:, self._left_gripper_dofs[0]] = -0.5 * self._left_open_driver
            q[:, self._left_gripper_dofs[1]] = 0.5 * self._left_open_driver

    def _reset_control_targets(self, env_ids) -> None:
        ids = self._env_indices_t(env_ids)
        joint_coord_ids = self._joint_coord_ids_t.index_select(0, ids)
        target_q = self._nominal_control_q.index_select(0, ids)
        self._last_control_q[ids] = target_q
        self._control_joint_target_pos_t.scatter_(0, joint_coord_ids.reshape(-1), target_q.reshape(-1))

    def _gripper_joint_step_limit(self) -> float:
        max_step = float(self.cfg.max_gripper_joint_step)
        max_velocity = float(self.cfg.max_gripper_joint_velocity)
        if max_velocity <= 0.0:
            return max_step

        step_dt = getattr(self._env, "step_dt", None)
        if step_dt is None:
            step_dt = float(self._env.cfg.sim.dt) * int(self._env.cfg.decimation)
        return min(max_step, max_velocity * float(step_dt))

    def _env_indices_t(self, env_ids) -> torch.Tensor:
        if env_ids is None:
            return self._all_env_ids_t
        if isinstance(env_ids, slice):
            start, stop, step = env_ids.indices(self.num_envs)
            if start == 0 and stop == self.num_envs and step == 1:
                return self._all_env_ids_t
            return torch.arange(start, stop, step, device=self.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long).flatten()
        if isinstance(env_ids, np.ndarray):
            return torch.as_tensor(env_ids.reshape(-1), device=self.device, dtype=torch.long)
        if isinstance(env_ids, int):
            return torch.as_tensor([env_ids], device=self.device, dtype=torch.long)
        return torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long)


def _resolve_body_ids(labels: list[str], short_name: str, num_envs: int) -> list[int]:
    suffix = "/" + short_name
    matches = [idx for idx, label in enumerate(labels) if label == short_name or label.endswith(suffix)]
    if len(matches) < num_envs:
        raise RuntimeError(f"Expected at least {num_envs} bodies named {short_name!r}, found {matches}.")
    return matches[:num_envs]


def _gripper_driver_targets(model, driver_dofs: list[int]) -> tuple[float, float]:
    if not driver_dofs:
        return 0.0, 0.0
    dof = driver_dofs[0]
    lower = float(model.joint_limit_lower.numpy()[dof])
    upper = float(model.joint_limit_upper.numpy()[dof])
    open_target = 0.5 * upper
    closed_target = 2.0 * 0.0036
    return max(lower, min(upper, open_target)), max(lower, min(upper, closed_target))


def _first_index(indices: list[int]) -> int | None:
    return int(indices[0]) if indices else None


def _right_arm_ik_coord_ids(model) -> list[int]:
    """Return right-arm joint coordinates controlled by waterhose IK."""
    coord_ids: list[int] = []
    joint_q_start = model.joint_q_start.numpy()
    for joint_id, label in enumerate(model.joint_label):
        short_label = str(label).rsplit("/", 1)[-1]
        if not short_label.startswith("right_arm_joint_"):
            continue
        coord_id = int(joint_q_start[joint_id])
        if 0 <= coord_id < int(model.joint_coord_count):
            coord_ids.append(coord_id)
    if not coord_ids:
        raise RuntimeError("No right-arm joint coordinates were found for waterhose Newton IK.")
    return sorted(set(coord_ids))


def _torch_quat_inverse(quat_xyzw: torch.Tensor) -> torch.Tensor:
    result = quat_xyzw.clone()
    result[..., :3] = -result[..., :3]
    return result


def _torch_quat_multiply(left_xyzw: torch.Tensor, right_xyzw: torch.Tensor) -> torch.Tensor:
    left_xyz = left_xyzw[..., :3]
    right_xyz = right_xyzw[..., :3]
    left_w = left_xyzw[..., 3:4]
    right_w = right_xyzw[..., 3:4]
    xyz = left_w * right_xyz + right_w * left_xyz + torch.linalg.cross(left_xyz, right_xyz, dim=-1)
    w = left_w * right_w - torch.sum(left_xyz * right_xyz, dim=-1, keepdim=True)
    return torch.cat((xyz, w), dim=-1)


def _torch_quat_normalize(quat_xyzw: torch.Tensor) -> torch.Tensor:
    return quat_xyzw / torch.linalg.vector_norm(quat_xyzw, dim=-1, keepdim=True).clamp_min(1.0e-12)


def _torch_quat_rotate(quat_xyzw: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat_xyz = quat_xyzw[..., :3]
    quat_w = quat_xyzw[..., 3:4]
    t = 2.0 * torch.linalg.cross(quat_xyz, vec, dim=-1)
    return vec + quat_w * t + torch.linalg.cross(quat_xyz, t, dim=-1)


def _torch_quat_from_axis_angle(axis_angle: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.vector_norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / angle.clamp_min(1.0e-12)
    half_angle = 0.5 * angle
    quat = torch.cat((axis * torch.sin(half_angle), torch.cos(half_angle)), dim=-1)
    identity = torch.zeros_like(quat)
    identity[..., 3] = 1.0
    return torch.where(angle > 1.0e-8, quat, identity)


def _torch_integrate_axis_angle_world(quat_xyzw: torch.Tensor, axis_angle: torch.Tensor) -> torch.Tensor:
    delta = _torch_quat_from_axis_angle(axis_angle)
    return _torch_quat_normalize(_torch_quat_multiply(delta, quat_xyzw))


def _torch_integrate_axis_angle_eef(quat_xyzw: torch.Tensor, axis_angle: torch.Tensor) -> torch.Tensor:
    delta = _torch_quat_from_axis_angle(axis_angle)
    return _torch_quat_normalize(_torch_quat_multiply(quat_xyzw, delta))


def _torch_axis_angle_between_world(curr_xyzw: torch.Tensor, target_xyzw: torch.Tensor) -> torch.Tensor:
    delta = _torch_quat_multiply(target_xyzw, _torch_quat_inverse(curr_xyzw))
    return _torch_axis_angle_from_quat(delta)


def _torch_axis_angle_between_eef(curr_xyzw: torch.Tensor, target_xyzw: torch.Tensor) -> torch.Tensor:
    delta = _torch_quat_multiply(_torch_quat_inverse(curr_xyzw), target_xyzw)
    return _torch_axis_angle_from_quat(delta)


def _torch_axis_angle_from_quat(delta_xyzw: torch.Tensor) -> torch.Tensor:
    delta = _torch_quat_normalize(delta_xyzw)
    xyz = delta[..., :3]
    xyz_norm = torch.linalg.vector_norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(xyz_norm, delta[..., 3:4])
    axis = xyz / xyz_norm.clamp_min(1.0e-12)
    return torch.where(angle > 1.0e-8, axis * angle, torch.zeros_like(xyz))
