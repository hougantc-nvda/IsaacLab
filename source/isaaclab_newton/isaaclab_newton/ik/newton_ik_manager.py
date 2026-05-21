# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import newton.ik as ik
import torch
import warp as wp

from .newton_ik_manager_cfg import NewtonIKManagerCfg


@dataclass(frozen=True)
class NewtonIKPoseObjective:
    """Additional pose objective for Newton IK."""

    link_index: int
    link_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    link_offset_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    position_weight: float = 1.0
    rotation_weight: float = 1.0


class NewtonIKManager:
    """Batched wrapper around Newton's inverse-kinematics solver."""

    cfg: NewtonIKManagerCfg

    def __init__(
        self,
        cfg: NewtonIKManagerCfg,
        *,
        model,
        num_envs: int,
        device: str,
        link_index: int,
        link_offset_pos: tuple[float, float, float],
        link_offset_rot: tuple[float, float, float, float],
        extra_pose_objectives: list[NewtonIKPoseObjective] | None = None,
    ):
        self.cfg = cfg
        self.model = model
        self.num_envs = num_envs
        self.device = device
        self.num_coords = model.joint_coord_count

        self.target_positions = wp.zeros((num_envs,), dtype=wp.vec3, device=device)
        self.target_rotations = wp.array(
            [(0.0, 0.0, 0.0, 1.0)] * num_envs,
            dtype=wp.vec4,
            device=device,
        )

        self.position_objective = ik.IKObjectivePosition(
            link_index=link_index,
            link_offset=wp.vec3(*link_offset_pos),
            target_positions=self.target_positions,
            weight=cfg.position_weight,
        )
        self.rotation_objective = ik.IKObjectiveRotation(
            link_index=link_index,
            link_offset_rotation=wp.quat(*link_offset_rot),
            target_rotations=self.target_rotations,
            weight=cfg.rotation_weight,
        )
        solver_objectives = [self.position_objective, self.rotation_objective]
        self.extra_position_objectives = []
        self.extra_rotation_objectives = []
        for objective_cfg in extra_pose_objectives or []:
            target_positions = wp.zeros((num_envs,), dtype=wp.vec3, device=device)
            target_rotations = wp.array(
                [(0.0, 0.0, 0.0, 1.0)] * num_envs,
                dtype=wp.vec4,
                device=device,
            )
            position_objective = ik.IKObjectivePosition(
                link_index=objective_cfg.link_index,
                link_offset=wp.vec3(*objective_cfg.link_offset_pos),
                target_positions=target_positions,
                weight=objective_cfg.position_weight,
            )
            rotation_objective = ik.IKObjectiveRotation(
                link_index=objective_cfg.link_index,
                link_offset_rotation=wp.quat(*objective_cfg.link_offset_rot),
                target_rotations=target_rotations,
                weight=objective_cfg.rotation_weight,
            )
            self.extra_position_objectives.append(position_objective)
            self.extra_rotation_objectives.append(rotation_objective)
            solver_objectives.extend((position_objective, rotation_objective))

        self.joint_limit_objective = ik.IKObjectiveJointLimit(
            joint_limit_lower=model.joint_limit_lower,
            joint_limit_upper=model.joint_limit_upper,
            weight=cfg.joint_limit_weight,
        )
        solver_objectives.append(self.joint_limit_objective)

        self.joint_q_out = wp.zeros((num_envs, self.num_coords), dtype=wp.float32, device=device)
        self.solver = ik.IKSolver(
            model=model,
            n_problems=num_envs,
            objectives=solver_objectives,
            optimizer=ik.IKOptimizer(cfg.optimizer),
            jacobian_mode=ik.IKJacobianType(cfg.jacobian_mode),
            lambda_initial=cfg.lambda_initial,
        )

    @property
    def action_dim(self) -> int:
        """Dimension of the IK command expected by this manager."""
        if self.cfg.command_type == "position":
            return 3
        if self.cfg.command_type == "pose" and self.cfg.use_relative_mode:
            return 6
        if self.cfg.command_type == "pose":
            return 7
        raise ValueError(f"Unsupported Newton IK command type: {self.cfg.command_type}")

    def set_target_pose(self, target_pos_w: torch.Tensor, target_quat_w: torch.Tensor) -> None:
        """Update batched world-frame target poses for the Newton IK objective."""
        self.position_objective.set_target_positions(wp.from_torch(target_pos_w.contiguous(), dtype=wp.vec3))
        self.rotation_objective.set_target_rotations(wp.from_torch(target_quat_w.contiguous(), dtype=wp.vec4))

    def set_extra_target_pose(
        self, objective_index: int, target_pos_w: torch.Tensor, target_quat_w: torch.Tensor
    ) -> None:
        """Update one auxiliary batched target pose."""
        self.extra_position_objectives[objective_index].set_target_positions(
            wp.from_torch(target_pos_w.contiguous(), dtype=wp.vec3)
        )
        self.extra_rotation_objectives[objective_index].set_target_rotations(
            wp.from_torch(target_quat_w.contiguous(), dtype=wp.vec4)
        )

    def solve(self, joint_pos: torch.Tensor) -> torch.Tensor:
        """Solve IK from the provided batched joint-coordinate seed."""
        if joint_pos.shape != (self.num_envs, self.num_coords):
            raise ValueError(
                f"Expected joint seed shape {(self.num_envs, self.num_coords)}, got {tuple(joint_pos.shape)}."
            )
        joint_q_in = wp.from_torch(joint_pos.contiguous(), dtype=wp.float32)
        self.solver.step(
            joint_q_in,
            self.joint_q_out,
            iterations=self.cfg.iterations,
            step_size=self.cfg.step_size,
        )
        return wp.to_torch(self.joint_q_out)
