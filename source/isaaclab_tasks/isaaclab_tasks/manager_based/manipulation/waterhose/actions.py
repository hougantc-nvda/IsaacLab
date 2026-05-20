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
    ik_iterations: int = 12
    controller: NewtonIKManagerCfg = NewtonIKManagerCfg(
        command_type="pose",
        use_relative_mode=True,
        optimizer="lm",
        jacobian_mode="analytic",
        iterations=12,
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
        core.apply_viewer_forces(self._env.sim)
        state = core.NewtonManager.get_state_0()
        body_q = state.body_q.numpy()
        joint_q = state.joint_q.numpy()
        action_np = self._processed_actions.detach().cpu().numpy()
        target_np = self._control.joint_target_pos.numpy().astype(np.float32, copy=True)
        for env_id, joint_coord_ids in enumerate(self._scene_builder.robot_joint_coord_ids_by_env):
            ee_q = body_q[self._body_ids[env_id]]
            if self.cfg.accumulate_targets:
                target_pos, target_quat = self._accumulate_target_delta(
                    env_id, action_np[env_id, :6].astype(np.float64)
                )
            else:
                ee_quat = ee_q[3:].astype(np.float64)
                delta_pos, target_quat = self._target_delta(ee_quat, action_np[env_id, :6].astype(np.float64))
                target_pos = ee_q[:3].astype(np.float64) + delta_pos
            q = self._solve_env(env_id, joint_q[joint_coord_ids].astype(np.float32), target_pos, target_quat)
            self._set_gripper_targets(q, float(action_np[env_id, 6]))
            max_step = np.full_like(q, float(self.cfg.max_joint_step))
            if self._scene_builder.gripper_dofs:
                max_step[self._scene_builder.gripper_dofs] = self._gripper_joint_step_limit()
            q = self._last_control_q[env_id] + np.clip(q - self._last_control_q[env_id], -max_step, max_step)
            self._last_control_q[env_id] = q.astype(np.float32, copy=True)
            target_np[joint_coord_ids] = q
        core.wp.copy(
            self._control.joint_target_pos,
            core.wp.array(target_np, dtype=core.wp.float32, device=self._model.device),
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._reset_task_targets(env_ids)
        self._reset_control_targets(env_ids)

    def target_pose_to_action(
        self, target_pos: torch.Tensor, target_quat_xyzw: torch.Tensor, env_id: int = 0
    ) -> torch.Tensor:
        curr = core.NewtonManager.get_state_0().body_q.numpy()[self._body_ids[env_id]]
        curr_pos = torch.as_tensor(curr[:3], device=self.device, dtype=torch.float32)
        delta_pos = (target_pos.to(self.device) - curr_pos).detach().cpu().numpy().astype(np.float64)
        curr_quat = torch.as_tensor(curr[3:], device=self.device, dtype=torch.float32)
        curr_quat_np = curr_quat.detach().cpu().numpy().astype(np.float64)
        target_quat_np = target_quat_xyzw.detach().cpu().numpy().astype(np.float64)
        if self.cfg.command_frame == "eef":
            delta_pos = core._np_quat_rotate(core._np_quat_inverse(curr_quat_np), delta_pos)
        if self._rotation_frame == "eef":
            delta_axis_np = _axis_angle_between_eef(curr_quat_np, target_quat_np)
        else:
            delta_axis_np = _axis_angle_between_world(curr_quat_np, target_quat_np)
        action = torch.zeros(7, device=self.device)
        action[:3] = torch.as_tensor(delta_pos, device=self.device, dtype=torch.float32) / float(
            self.cfg.position_scale
        )
        action[3:6] = torch.as_tensor(delta_axis_np, device=self.device, dtype=torch.float32) / float(
            self.cfg.rotation_scale
        )
        return action.clamp(-1.0, 1.0)

    def action_to_target_pose(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state_np = core.NewtonManager.get_state_0().body_q.numpy()
        processed_position = actions[:, :3] * float(self.cfg.position_scale)
        processed_rotation = actions[:, 3:6] * float(self.cfg.rotation_scale)
        pos = []
        quat = []
        for env_id in range(actions.shape[0]):
            ee_q = state_np[self._body_ids[env_id]]
            ee_quat = ee_q[3:].astype(np.float64)
            delta_pos, quat_np = self._target_delta(
                ee_quat,
                torch.cat((processed_position[env_id], processed_rotation[env_id])).detach().cpu().numpy(),
            )
            pos.append(torch.as_tensor(ee_q[:3] + delta_pos, device=self.device, dtype=torch.float32))
            quat.append(torch.as_tensor(quat_np, device=self.device, dtype=torch.float32))
        return torch.stack(pos, dim=0), torch.stack(quat, dim=0)

    def _target_delta(self, ee_quat: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        delta_pos = action[:3]
        axis_angle = action[3:6]
        if self.cfg.command_frame == "eef":
            delta_pos = core._np_quat_rotate(ee_quat, delta_pos)
        if self._rotation_frame == "eef":
            target_quat = _integrate_axis_angle_eef(ee_quat, axis_angle)
        else:
            target_quat = _integrate_axis_angle_world(ee_quat, axis_angle)
        return delta_pos, target_quat

    def _seed_task_targets(self) -> None:
        body_q = core.NewtonManager.get_state_0().body_q.numpy()
        self._target_pos = np.zeros((self.num_envs, 3), dtype=np.float64)
        self._target_quat = np.zeros((self.num_envs, 4), dtype=np.float64)
        for env_id, body_id in enumerate(self._body_ids):
            q = body_q[body_id]
            self._target_pos[env_id] = q[:3].astype(np.float64)
            self._target_quat[env_id] = q[3:].astype(np.float64)

    def _reset_task_targets(self, env_ids) -> None:
        body_q = core.NewtonManager.get_state_0().body_q.numpy()
        for env_id in self._env_indices(env_ids):
            q = body_q[self._body_ids[env_id]]
            self._target_pos[env_id] = q[:3].astype(np.float64)
            self._target_quat[env_id] = q[3:].astype(np.float64)

    def _accumulate_target_delta(self, env_id: int, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        delta_pos, target_quat = self._target_delta(self._target_quat[env_id], action)
        self._target_pos[env_id] = self._target_pos[env_id] + delta_pos
        self._target_quat[env_id] = target_quat
        return self._target_pos[env_id], self._target_quat[env_id]

    def _setup_ik(self) -> None:
        link_index = _resolve_body_ids(self._single_robot_model.body_label, core.RIGHT_EE, 1)[0]
        self.cfg.controller.iterations = int(self.cfg.ik_iterations)
        self._ik_control_coord_ids = _controlled_ik_coord_ids(self._single_robot_model)
        self._ik_manager = NewtonIKManager(
            self.cfg.controller,
            model=self._single_robot_model,
            num_envs=1,
            device=self.device,
            link_index=link_index,
            link_offset_pos=(0.0, 0.0, 0.0),
            link_offset_rot=(0.0, 0.0, 0.0, 1.0),
        )

    def _seed_control_targets(self) -> None:
        initial = self._single_robot_model.joint_q.numpy().astype(np.float32, copy=False)
        target_np = self._control.joint_target_pos.numpy().astype(np.float32, copy=True)
        self._last_control_q = np.tile(initial, (self.num_envs, 1)).astype(np.float32)
        for joint_coord_ids in self._scene_builder.robot_joint_coord_ids_by_env:
            target_np[joint_coord_ids] = initial
        core.wp.copy(
            self._control.joint_target_pos,
            core.wp.array(target_np, dtype=core.wp.float32, device=self._model.device),
        )

    def _solve_env(
        self, env_id: int, current_q: np.ndarray, target_pos: np.ndarray, target_quat: np.ndarray
    ) -> np.ndarray:
        origin = self._env_origin(env_id)
        local_target_pos = target_pos - origin
        target_pos_t = torch.as_tensor(local_target_pos, device=self.device, dtype=torch.float32).reshape(1, 3)
        target_quat_t = torch.as_tensor(target_quat, device=self.device, dtype=torch.float32).reshape(1, 4)
        seed_t = torch.as_tensor(current_q, device=self.device, dtype=torch.float32).reshape(
            1, self._single_robot_model.joint_coord_count
        )
        self._ik_manager.set_target_pose(target_pos_t, target_quat_t)
        solved_q = self._ik_manager.solve(seed_t)[0].detach().cpu().numpy().astype(np.float32, copy=True)
        q = current_q.astype(np.float32, copy=True)
        q[self._ik_control_coord_ids] = solved_q[self._ik_control_coord_ids]
        return q

    def _set_gripper_targets(self, q: np.ndarray, gripper_action: float) -> None:
        right_alpha = 1.0 if gripper_action > 0.0 else 0.0
        right_driver = (1.0 - right_alpha) * self._right_open_driver + right_alpha * self._right_closed_driver
        _set_gripper_side(
            q, self._scene_builder.right_gripper_driver_dofs, self._scene_builder.right_gripper_dofs, right_driver
        )
        _set_gripper_side(
            q,
            self._scene_builder.left_gripper_driver_dofs,
            self._scene_builder.left_gripper_dofs,
            self._left_open_driver,
        )

    def _reset_control_targets(self, env_ids) -> None:
        joint_q = core.NewtonManager.get_state_0().joint_q.numpy()
        target_np = self._control.joint_target_pos.numpy().astype(np.float32, copy=True)
        for env_id in self._env_indices(env_ids):
            joint_coord_ids = self._scene_builder.robot_joint_coord_ids_by_env[env_id]
            q = joint_q[joint_coord_ids].astype(np.float32, copy=True)
            self._last_control_q[env_id] = q
            target_np[joint_coord_ids] = q
        core.wp.copy(
            self._control.joint_target_pos,
            core.wp.array(target_np, dtype=core.wp.float32, device=self._model.device),
        )

    def _gripper_joint_step_limit(self) -> float:
        max_step = float(self.cfg.max_gripper_joint_step)
        max_velocity = float(self.cfg.max_gripper_joint_velocity)
        if max_velocity <= 0.0:
            return max_step

        step_dt = getattr(self._env, "step_dt", None)
        if step_dt is None:
            step_dt = float(self._env.cfg.sim.dt) * int(self._env.cfg.decimation)
        return min(max_step, max_velocity * float(step_dt))

    def _env_indices(self, env_ids) -> list[int]:
        if isinstance(env_ids, slice):
            return list(range(*env_ids.indices(self.num_envs)))
        if isinstance(env_ids, torch.Tensor):
            return [int(env_id) for env_id in env_ids.detach().cpu().flatten().tolist()]
        if isinstance(env_ids, np.ndarray):
            return [int(env_id) for env_id in env_ids.reshape(-1).tolist()]
        if isinstance(env_ids, int):
            return [env_ids]
        return [int(env_id) for env_id in env_ids]

    def _env_origin(self, env_id: int) -> np.ndarray:
        return np.array([float(self._scene_builder.env_origins[env_id][i]) for i in range(3)], dtype=np.float64)


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


def _set_gripper_side(q: np.ndarray, driver_dofs: list[int], finger_dofs: list[int], driver_target: float) -> None:
    if driver_dofs:
        q[driver_dofs[0]] = driver_target
    if len(finger_dofs) >= 2:
        q[finger_dofs[0]] = -0.5 * driver_target
        q[finger_dofs[1]] = 0.5 * driver_target


def _controlled_ik_coord_ids(model) -> list[int]:
    """Return torso and right-arm joint coordinates controlled by waterhose IK."""
    coord_ids: list[int] = []
    joint_q_start = model.joint_q_start.numpy()
    for joint_id, label in enumerate(model.joint_label):
        short_label = str(label).rsplit("/", 1)[-1]
        if not (short_label.startswith("torso_joint_") or short_label.startswith("right_arm_joint_")):
            continue
        coord_id = int(joint_q_start[joint_id])
        if 0 <= coord_id < int(model.joint_coord_count):
            coord_ids.append(coord_id)
    if not coord_ids:
        raise RuntimeError("No torso/right-arm joint coordinates were found for waterhose Newton IK.")
    return sorted(set(coord_ids))


def _integrate_axis_angle_world(quat_xyzw: np.ndarray, axis_angle: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(axis_angle))
    if angle <= 1.0e-8:
        return quat_xyzw / max(np.linalg.norm(quat_xyzw), 1.0e-12)
    delta = core._np_quat_from_axis_angle(axis_angle / angle, angle)
    quat = core._np_quat_multiply(delta, quat_xyzw)
    return quat / max(np.linalg.norm(quat), 1.0e-12)


def _integrate_axis_angle_eef(quat_xyzw: np.ndarray, axis_angle: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(axis_angle))
    if angle <= 1.0e-8:
        return quat_xyzw / max(np.linalg.norm(quat_xyzw), 1.0e-12)
    delta = core._np_quat_from_axis_angle(axis_angle / angle, angle)
    quat = core._np_quat_multiply(quat_xyzw, delta)
    return quat / max(np.linalg.norm(quat), 1.0e-12)


def _axis_angle_between_world(curr_xyzw: np.ndarray, target_xyzw: np.ndarray) -> np.ndarray:
    delta = core._np_quat_multiply(target_xyzw, core._np_quat_inverse(curr_xyzw))
    return _axis_angle_from_quat(delta)


def _axis_angle_between_eef(curr_xyzw: np.ndarray, target_xyzw: np.ndarray) -> np.ndarray:
    delta = core._np_quat_multiply(core._np_quat_inverse(curr_xyzw), target_xyzw)
    return _axis_angle_from_quat(delta)


def _axis_angle_from_quat(delta: np.ndarray) -> np.ndarray:
    delta = delta / max(np.linalg.norm(delta), 1.0e-12)
    angle = 2.0 * np.arctan2(np.linalg.norm(delta[:3]), delta[3])
    if angle <= 1.0e-8:
        return np.zeros(3, dtype=np.float64)
    axis = delta[:3] / max(np.linalg.norm(delta[:3]), 1.0e-12)
    return axis * angle
