# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence

import newton
import newton.ik as ik
import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics import (
    NewtonManager,
)
from newton import JointTargetMode, eval_fk
from newton.solvers import SolverImplicitMPM, SolverMuJoCo

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from .ur10_particle_scoop_env_cfg import UR10ParticleScoopEnvCfg

logger = logging.getLogger(__name__)


class PolicyObservationSpheres:
    """Render the policy's compact particle observations as Newton sphere markers."""

    def __init__(self):
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/PolicyObservations",
            markers={
                "grid_cell": sim_utils.SphereCfg(
                    radius=0.006,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15)),
                ),
                "height_cell": sim_utils.SphereCfg(
                    radius=0.012,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 1.0)),
                ),
                "dense_cell": sim_utils.SphereCfg(
                    radius=0.014,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.55, 0.10)),
                ),
                "centroid": sim_utils.SphereCfg(
                    radius=0.035,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.0)),
                ),
                "paddle": sim_utils.SphereCfg(
                    radius=0.025,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 1.0)),
                ),
                "bin": sim_utils.SphereCfg(
                    radius=0.025,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 1.0, 0.25)),
                ),
                "mouth": sim_utils.SphereCfg(
                    radius=0.02,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.15, 0.15)),
                ),
                "in_bin": sim_utils.SphereCfg(
                    radius=0.01,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 1.0, 0.1)),
                ),
                "spilled": sim_utils.SphereCfg(
                    radius=0.012,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.05, 0.05)),
                ),
                "pile_center": sim_utils.SphereCfg(
                    radius=0.025,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.2, 1.0)),
                ),
            },
        )
        self._markers = VisualizationMarkers(marker_cfg)
        self._markers.set_visibility(False)
        self._enabled = False

    def set_enabled(self, enabled: bool) -> None:
        if enabled == self._enabled:
            return
        self._enabled = enabled
        self._markers.set_visibility(enabled)

    def update(
        self,
        env: UR10ParticleScoopEnv,
        heightmap: torch.Tensor,
        density: torch.Tensor,
        paddle_corners: torch.Tensor,
        centroid: torch.Tensor,
        particle_pos: torch.Tensor,
    ) -> None:
        show_obs = _viewer_flag_enabled(env, "show_policy_observations")
        show_reward_debug = _viewer_flag_enabled(env, "show_reward_debug")
        show_curriculum_debug = _viewer_flag_enabled(env, "show_curriculum_debug")
        self.set_enabled(show_obs or show_reward_debug or show_curriculum_debug)
        if not self._enabled:
            return

        env_id = 0
        origin = env.scene.env_origins[env_id]
        map_size = env.cfg.heightmap_size
        grid_y, grid_x = torch.meshgrid(
            torch.arange(map_size, device=env.device),
            torch.arange(map_size, device=env.device),
            indexing="ij",
        )
        cell_x = grid_x.reshape(-1).float()
        cell_y = grid_y.reshape(-1).float()
        cell_height = heightmap[env_id].reshape(-1)
        cell_density = density[env_id].reshape(-1)
        cell_pos = torch.stack(
            (
                env._heightmap_x_min + (cell_x + 0.5) * env._heightmap_x_range / map_size,
                env._heightmap_y_min + (cell_y + 0.5) * env._heightmap_y_range / map_size,
                env._heightmap_z_min + cell_height * env.cfg.heightmap_z_range + 0.02,
            ),
            dim=-1,
        )
        translations = cell_pos + origin
        marker_indices = torch.zeros_like(cell_height, dtype=torch.int32)
        marker_indices = torch.where(cell_height > 1.0e-3, torch.ones_like(marker_indices), marker_indices)
        marker_indices = torch.where(cell_density > 0.35, torch.full_like(marker_indices, 2), marker_indices)

        feature_pos = torch.stack(
            (
                centroid[env_id] + origin,
                env._bin_target + origin,
                env._bin_mouth_center + origin,
            ),
            dim=0,
        )
        corner_pos = paddle_corners[env_id] + origin
        translations = torch.cat((translations, feature_pos[:1], corner_pos, feature_pos[1:]), dim=0)
        feature_indices = torch.tensor((3, 4, 4, 4, 4, 5, 6), dtype=torch.int32, device=env.device)
        marker_indices = torch.cat((marker_indices, feature_indices), dim=0)
        if show_reward_debug:
            env_particles = particle_pos[env_id]
            in_bin_particles = env_particles[env._particles_in_bin(particle_pos)[env_id]]
            spilled_particles = env_particles[env._particles_spilled(particle_pos)[env_id]]
            debug_translations = []
            debug_indices = []
            if in_bin_particles.numel() > 0:
                debug_translations.append(in_bin_particles + origin)
                debug_indices.append(torch.full((in_bin_particles.shape[0],), 7, dtype=torch.int32, device=env.device))
            if spilled_particles.numel() > 0:
                debug_translations.append(spilled_particles + origin)
                debug_indices.append(torch.full((spilled_particles.shape[0],), 8, dtype=torch.int32, device=env.device))
            if debug_translations:
                translations = torch.cat((translations, *debug_translations), dim=0)
                marker_indices = torch.cat((marker_indices, *debug_indices), dim=0)
        if show_curriculum_debug:
            pile_marker = env._current_pile_center[env_id : env_id + 1] + origin
            translations = torch.cat((translations, pile_marker), dim=0)
            marker_indices = torch.cat(
                (marker_indices, torch.full((1,), 9, dtype=torch.int32, device=env.device)), dim=0
            )
        self._markers.visualize(translations=translations, marker_indices=marker_indices)


def _viewer_flag_enabled(env: UR10ParticleScoopEnv, flag_name: str) -> bool:
    for visualizer in env.sim.visualizers:
        viewer = getattr(visualizer, "_viewer", None)
        if viewer is not None and getattr(viewer, flag_name, False):
            return True
    return False


class UR10ParticleScoopEnv(DirectRLEnv):
    """Pure Newton UR10 + MPM particle scooping task."""

    cfg: UR10ParticleScoopEnvCfg

    def __init__(self, cfg: UR10ParticleScoopEnvCfg, render_mode: str | None = None, **kwargs):
        self._joint_q_ids_list: list[list[int]] = []
        self._joint_qd_ids_list: list[list[int]] = []
        self._particle_ids_list: list[list[int]] = []
        self._ee_body_ids_list: list[int] = []
        super().__init__(cfg, render_mode, **kwargs)

        self._joint_q_ids = torch.tensor(self._joint_q_ids_list, device=self.device, dtype=torch.long)
        self._joint_qd_ids = torch.tensor(self._joint_qd_ids_list, device=self.device, dtype=torch.long)
        self._particle_ids = torch.tensor(self._particle_ids_list, device=self.device, dtype=torch.long)
        self._ee_body_ids = torch.tensor(self._ee_body_ids_list, device=self.device, dtype=torch.long)
        self._particle_count = int(self._particle_ids.shape[1])
        self._mpm_solver = self._create_mpm_solver()
        self._mpm_graph = self._capture_mpm_graph()

        state = NewtonManager.get_state_0()
        model = NewtonManager.get_model()
        self._default_joint_q = wp.to_torch(state.joint_q)[self._joint_q_ids].clone()
        self._default_joint_qd = wp.to_torch(state.joint_qd)[self._joint_qd_ids].clone()
        joint_lower = wp.to_torch(model.joint_limit_lower)[self._joint_qd_ids].clone()
        joint_upper = wp.to_torch(model.joint_limit_upper)[self._joint_qd_ids].clone()
        finite_limits = (
            torch.isfinite(joint_lower) & torch.isfinite(joint_upper) & ((joint_upper - joint_lower) > 1.0e-3)
        )
        reasonable_limits = finite_limits & ((joint_upper - joint_lower) < 100.0)
        self._joint_lower = torch.where(reasonable_limits, joint_lower, torch.full_like(joint_lower, -2.0 * torch.pi))
        self._joint_upper = torch.where(reasonable_limits, joint_upper, torch.full_like(joint_upper, 2.0 * torch.pi))
        self._joint_center = 0.5 * (self._joint_lower + self._joint_upper)
        self._joint_half_range = torch.clamp(0.5 * (self._joint_upper - self._joint_lower), min=1.0e-6)
        self._create_ik_solver()
        self._default_particle_q = wp.to_torch(state.particle_q)[self._particle_ids].clone()
        self._default_particle_e = self._default_particle_q - self.scene.env_origins[:, None, :]
        self._joint_targets = self._default_joint_q.clone()
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._bin_center = torch.tensor(self.cfg.bin_center, device=self.device)
        self._bin_half_extents = torch.tensor(self.cfg.bin_inner_half_extents, device=self.device)
        self._bin_lower = self._bin_center - self._bin_half_extents
        self._bin_upper = self._bin_center + self._bin_half_extents
        self._bin_lower[2] = self._bin_lower[2] + self.cfg.bin_particle_min_height
        self._bin_target = self._bin_center.clone()
        self._bin_target[2] = self._bin_center[2]
        self._bin_mouth_center = self._bin_center.clone()
        self._bin_mouth_center[0] = self._bin_lower[0]
        self._bin_mouth_center[2] = (
            self.cfg.bin_center[2] - self.cfg.bin_inner_half_extents[2] + self.cfg.bin_front_wall_height
        )
        self._workspace_lower = torch.tensor(
            (
                self.cfg.table_center[0] - 0.5 * self.cfg.table_size[0] - 0.10,
                self.cfg.table_center[1] - 0.5 * self.cfg.table_size[1] - 0.15,
                min(self.cfg.heightmap_z_min, float(self._bin_lower[2])) - 0.05,
            ),
            device=self.device,
        )
        self._workspace_upper = torch.tensor(
            (
                self._bin_upper[0].item() + 0.15,
                self.cfg.table_center[1] + 0.5 * self.cfg.table_size[1] + 0.15,
                max(self.cfg.table_top_z + self.cfg.heightmap_z_range, float(self._bin_upper[2]) + 0.10),
            ),
            device=self.device,
        )
        self._paddle_workspace_lower = torch.tensor(
            (
                self.cfg.paddle_workspace_x_bounds[0],
                self.cfg.paddle_workspace_y_bounds[0],
                self.cfg.paddle_center_min_height,
            ),
            device=self.device,
        )
        self._paddle_workspace_upper = torch.tensor(
            (
                self.cfg.paddle_workspace_x_bounds[1],
                self.cfg.paddle_workspace_y_bounds[1],
                self.cfg.paddle_max_height,
            ),
            device=self.device,
        )
        self._default_pile_center = 0.5 * (
            torch.tensor(self.cfg.pile_lo, device=self.device) + torch.tensor(self.cfg.pile_hi, device=self.device)
        )
        self._default_pile_half_extents = 0.5 * (
            torch.tensor(self.cfg.pile_hi, device=self.device) - torch.tensor(self.cfg.pile_lo, device=self.device)
        )
        self._default_particle_offsets = self._default_particle_e - self._default_pile_center
        self._current_pile_center = self._default_pile_center.unsqueeze(0).repeat(self.num_envs, 1)
        self._current_pile_half_extents = self._default_pile_half_extents.unsqueeze(0).repeat(self.num_envs, 1)
        self._table_xy_lower = torch.tensor(
            (
                self.cfg.table_center[0] - 0.5 * self.cfg.table_size[0],
                self.cfg.table_center[1] - 0.5 * self.cfg.table_size[1],
            ),
            device=self.device,
        )
        self._table_xy_upper = torch.tensor(
            (
                self.cfg.table_center[0] + 0.5 * self.cfg.table_size[0],
                self.cfg.table_center[1] + 0.5 * self.cfg.table_size[1],
            ),
            device=self.device,
        )
        self._progress_target_x = float(self._bin_mouth_center[0])
        self._heightmap_x_min = float(self.cfg.heightmap_x_bounds[0])
        self._heightmap_x_range = float(self.cfg.heightmap_x_bounds[1] - self.cfg.heightmap_x_bounds[0])
        self._heightmap_y_min = float(self.cfg.heightmap_y_bounds[0])
        self._heightmap_y_range = float(self.cfg.heightmap_y_bounds[1] - self.cfg.heightmap_y_bounds[0])
        self._heightmap_z_min = float(self.cfg.heightmap_z_min)
        self._heightmap_env_offsets = (
            torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            * self.cfg.heightmap_size
            * self.cfg.heightmap_size
        ).unsqueeze(1)
        self._paddle_center_offset = torch.tensor(self.cfg.paddle_ee_offset, device=self.device)
        paddle_x, paddle_y, _ = self.cfg.paddle_size
        self._paddle_corner_offsets = (
            torch.tensor(
                (
                    (-0.5 * paddle_x, -0.5 * paddle_y, 0.0),
                    (-0.5 * paddle_x, 0.5 * paddle_y, 0.0),
                    (0.5 * paddle_x, -0.5 * paddle_y, 0.0),
                    (0.5 * paddle_x, 0.5 * paddle_y, 0.0),
                ),
                device=self.device,
            )
            + self._paddle_center_offset
        )
        self._paddle_half_size = 0.5 * torch.tensor(self.cfg.paddle_size, device=self.device)
        self._previous_bin_count = torch.zeros(self.num_envs, device=self.device)
        self._previous_bin_fraction = torch.zeros(self.num_envs, device=self.device)
        self._previous_particle_progress = torch.zeros(self.num_envs, device=self.device)
        self._previous_centroid_progress = torch.zeros(self.num_envs, device=self.device)
        self._previous_bin_proximity = torch.zeros(self.num_envs, device=self.device)
        self._previous_paddle_pos = self._paddle_pos_e().clone()
        self._target_paddle_pos = self._previous_paddle_pos.clone()
        self._previous_particle_centroid = self._scoopable_particle_centroid(self._particle_pos_e()).clone()
        self._previous_actions = torch.zeros_like(self._actions)
        self._episode_succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._curriculum_stage = 0
        self._curriculum_success_ema = 0.0
        self._curriculum_bin_fraction_ema = 0.0
        self._curriculum_resets_in_stage = 0
        self._last_bin_fraction = torch.zeros(self.num_envs, device=self.device)
        self._last_spill_fraction = torch.zeros(self.num_envs, device=self.device)
        self._last_particle_progress = torch.zeros(self.num_envs, device=self.device)
        self._last_centroid_progress = torch.zeros(self.num_envs, device=self.device)
        self._last_bin_proximity = torch.zeros(self.num_envs, device=self.device)
        self._last_mouth_entry = torch.zeros(self.num_envs, device=self.device)
        self._last_paddle_speed = torch.zeros(self.num_envs, device=self.device)
        self._last_action_rate = torch.zeros(self.num_envs, device=self.device)
        self._last_paddle_bin_proximity = torch.zeros(self.num_envs, device=self.device)
        self._last_paddle_orientation = torch.zeros(self.num_envs, device=self.device)
        self._last_paddle_setup = torch.zeros(self.num_envs, device=self.device)
        self._last_paddle_contact = torch.zeros(self.num_envs, device=self.device)
        self._last_paddle_push_speed = torch.zeros(self.num_envs, device=self.device)
        self._last_particle_push_speed = torch.zeros(self.num_envs, device=self.device)
        self._last_paddle_retreat_speed = torch.zeros(self.num_envs, device=self.device)
        self._last_paddle_velocity = torch.zeros(self.num_envs, 3, device=self.device)
        self._last_particle_centroid_velocity = torch.zeros(self.num_envs, 3, device=self.device)
        self._nonfinite_state = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._nonfinite_particles = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._nonfinite_joint_q = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._nonfinite_joint_qd = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._nonfinite_body = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_nonfinite_observation = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_reset_ik_failure = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_reset_ik_position_error = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums = {
            "bin_fraction": torch.zeros(self.num_envs, device=self.device),
            "delta_bin_fraction": torch.zeros(self.num_envs, device=self.device),
            "particle_progress": torch.zeros(self.num_envs, device=self.device),
            "centroid_progress": torch.zeros(self.num_envs, device=self.device),
            "bin_proximity": torch.zeros(self.num_envs, device=self.device),
            "mouth_entry": torch.zeros(self.num_envs, device=self.device),
            "spill_penalty": torch.zeros(self.num_envs, device=self.device),
            "paddle_proximity": torch.zeros(self.num_envs, device=self.device),
            "paddle_bin_proximity": torch.zeros(self.num_envs, device=self.device),
            "paddle_orientation": torch.zeros(self.num_envs, device=self.device),
            "paddle_setup": torch.zeros(self.num_envs, device=self.device),
            "paddle_contact": torch.zeros(self.num_envs, device=self.device),
            "paddle_push_velocity": torch.zeros(self.num_envs, device=self.device),
            "particle_push_velocity": torch.zeros(self.num_envs, device=self.device),
            "paddle_retreat_penalty": torch.zeros(self.num_envs, device=self.device),
            "paddle_low_penalty": torch.zeros(self.num_envs, device=self.device),
            "paddle_speed_penalty": torch.zeros(self.num_envs, device=self.device),
            "action_penalty": torch.zeros(self.num_envs, device=self.device),
            "action_rate_penalty": torch.zeros(self.num_envs, device=self.device),
            "joint_velocity_penalty": torch.zeros(self.num_envs, device=self.device),
            "success_bonus": torch.zeros(self.num_envs, device=self.device),
            "nonfinite_penalty": torch.zeros(self.num_envs, device=self.device),
        }
        self._policy_observation_spheres = PolicyObservationSpheres()
        self._configure_newton_viewer()

    def step(self, action: torch.Tensor) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        action = action.to(self.device)
        if self.cfg.action_noise_model:
            action = self._action_noise_model(action)

        self._pre_physics_step(action)
        is_rendering = self.sim.is_rendering

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self._step_mpm()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render(skip_app_pumping=not self.render_enabled)
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1).int()
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            if self.render_enabled and is_rendering and self.has_rtx_sensors and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

        if self.cfg.events and "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.obs_buf = self._get_observations()
        if self.cfg.observation_noise_model:
            self.obs_buf["proprio"] = self._observation_noise_model(self.obs_buf["proprio"])

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _create_mpm_solver(self) -> SolverImplicitMPM:
        model = NewtonManager.get_model()
        state = NewtonManager.get_state_0()
        mpm_cfg = SolverImplicitMPM.Config()
        mpm_cfg.voxel_size = self.cfg.voxel_size
        mpm_cfg.grid_type = "fixed"
        mpm_cfg.grid_padding = self.cfg.mpm_grid_padding
        mpm_cfg.max_active_cell_count = self.cfg.mpm_max_active_cell_count
        mpm_cfg.strain_basis = "P0"
        mpm_cfg.transfer_scheme = "pic"
        mpm_cfg.max_iterations = self.cfg.mpm_iterations
        mpm_cfg.critical_fraction = 0.0
        mpm_cfg.air_drag = 1.0
        mpm_cfg.collider_velocity_mode = "backward"
        mpm_cfg.solver = "gauss-seidel"
        mpm_solver = SolverImplicitMPM(model, mpm_cfg)
        mpm_solver.setup_collider(body_mass=wp.zeros_like(model.body_mass), body_q=state.body_q)
        return mpm_solver

    def _capture_mpm_graph(self):
        if not wp.get_device().is_cuda or self._mpm_solver.grid_type != "fixed":
            return None
        try:
            with wp.ScopedCapture() as capture:
                self._simulate_mpm_step()
            return capture.graph
        except RuntimeError as exc:
            logger.warning("UR10 MPM CUDA graph capture failed; using eager MPM stepping: %s", exc)
            return None

    def _step_mpm(self) -> None:
        if self._mpm_graph is not None:
            wp.capture_launch(self._mpm_graph)
        else:
            self._simulate_mpm_step()

    def _simulate_mpm_step(self) -> None:
        state = NewtonManager.get_state_0()
        self._mpm_solver.step(state, state, control=None, contacts=None, dt=self.physics_dt)

    @staticmethod
    def _replace_nonfinite(values: torch.Tensor, fallback: torch.Tensor | float = 0.0) -> torch.Tensor:
        fallback_tensor = torch.as_tensor(fallback, device=values.device, dtype=values.dtype)
        fallback_tensor = torch.broadcast_to(fallback_tensor, values.shape)
        return torch.where(torch.isfinite(values), values, fallback_tensor)

    @staticmethod
    def _sanitize_quat(quat: torch.Tensor) -> torch.Tensor:
        identity = torch.zeros_like(quat)
        identity[..., 3] = 1.0
        finite = torch.isfinite(quat).all(dim=-1, keepdim=True)
        norm = torch.linalg.norm(torch.nan_to_num(quat), dim=-1, keepdim=True)
        valid = finite & (norm > 1.0e-8)
        return torch.where(valid, quat / torch.clamp(norm, min=1.0e-8), identity)

    def _create_ik_solver(self) -> None:
        ik_builder = NewtonManager.create_builder()
        SolverMuJoCo.register_custom_attributes(ik_builder)
        ik_builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.1,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        ik_builder.add_urdf(
            self.cfg.ur10_urdf_path,
            xform=wp.transform(wp.vec3(*self.cfg.robot_base_pos), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=False,
            ignore_inertial_definitions=False,
        )
        self._configure_ur10_joints(ik_builder)
        ik_ee_body = self._find_body(ik_builder, self.cfg.ee_body_name)
        ik_arm_q_ids, _ = self._resolve_arm_joint_ids(ik_builder)
        if len(ik_arm_q_ids) != self.cfg.arm_dof_count:
            raise RuntimeError(f"Expected {self.cfg.arm_dof_count} UR10 IK DOFs, found {len(ik_arm_q_ids)}.")

        ik_device = NewtonManager.get_model().device
        self._ik_model = ik_builder.finalize(device=ik_device)
        self._ik_target_positions = wp.zeros(self.num_envs, dtype=wp.vec3, device=ik_device)
        self._ik_target_rotations = wp.zeros(self.num_envs, dtype=wp.vec4, device=ik_device)
        self._ik_joint_q_in = wp.zeros(
            (self.num_envs, self._ik_model.joint_coord_count), dtype=wp.float32, device=ik_device
        )
        self._ik_joint_q_out = wp.zeros_like(self._ik_joint_q_in)
        self._ik_target_positions_t = wp.to_torch(self._ik_target_positions)
        self._ik_target_rotations_t = wp.to_torch(self._ik_target_rotations)
        self._ik_joint_q_in_t = wp.to_torch(self._ik_joint_q_in)
        self._ik_joint_q_out_t = wp.to_torch(self._ik_joint_q_out)
        self._ik_default_joint_q = wp.to_torch(self._ik_model.joint_q).clone()
        self._ik_arm_q_ids = torch.tensor(ik_arm_q_ids, device=self._ik_joint_q_in_t.device, dtype=torch.long)

        position_objective = ik.IKObjectivePosition(
            link_index=ik_ee_body,
            link_offset=wp.vec3(*self.cfg.paddle_ee_offset),
            target_positions=self._ik_target_positions,
            weight=self.cfg.ik_position_weight,
        )
        rotation_objective = ik.IKObjectiveRotation(
            link_index=ik_ee_body,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=self._ik_target_rotations,
            weight=self.cfg.ik_rotation_weight,
        )
        joint_limit_objective = ik.IKObjectiveJointLimit(
            self._ik_model.joint_limit_lower,
            self._ik_model.joint_limit_upper,
            weight=self.cfg.ik_joint_limit_weight,
        )
        self._ik_solver = ik.IKSolver(
            model=self._ik_model,
            n_problems=self.num_envs,
            objectives=[position_objective, rotation_objective, joint_limit_objective],
            lambda_initial=self.cfg.ik_lambda_initial,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def _setup_scene(self) -> None:
        builder = self._build_newton_model()
        NewtonManager._num_envs = self.scene.num_envs
        NewtonManager.set_builder(builder)

    def _build_newton_model(self) -> newton.ModelBuilder:
        proto, meta = self._build_world_proto()
        builder = NewtonManager.create_builder()
        SolverMuJoCo.register_custom_attributes(builder)
        SolverImplicitMPM.register_custom_attributes(builder)

        for env_id in range(self.scene.num_envs):
            origin = self.scene.env_origins[env_id].detach().cpu().tolist()
            builder.begin_world(label=f"env_{env_id}")
            body_offset = builder.body_count
            particle_offset = builder.particle_count
            joint_q_offset = builder.joint_coord_count
            joint_qd_offset = builder.joint_dof_count

            builder.add_builder(proto, xform=wp.transform(wp.vec3(*origin), wp.quat_identity()))

            particle_range = list(range(particle_offset, particle_offset + meta["particle_count"]))
            self._particle_ids_list.append(particle_range)
            self._joint_q_ids_list.append([joint_q_offset + idx for idx in meta["arm_q_ids"]])
            self._joint_qd_ids_list.append([joint_qd_offset + idx for idx in meta["arm_qd_ids"]])
            self._ee_body_ids_list.append(body_offset + int(meta["ee_body"]))
            builder.end_world()

        return builder

    def _build_world_proto(self) -> tuple[newton.ModelBuilder, dict[str, object]]:
        proto = NewtonManager.create_builder()
        SolverMuJoCo.register_custom_attributes(proto)
        SolverImplicitMPM.register_custom_attributes(proto)
        proto.default_shape_cfg.mu = 0.75
        proto.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.1,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )

        proto.add_urdf(
            self.cfg.ur10_urdf_path,
            xform=wp.transform(wp.vec3(*self.cfg.robot_base_pos), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=False,
            ignore_inertial_definitions=False,
        )
        self._configure_ur10_joints(proto)
        self._disable_robot_particle_collisions(proto)

        ee_body = self._find_body(proto, self.cfg.ee_body_name)
        paddle_shapes = self._add_paddle_pad(proto, ee_body)
        workspace_shapes = self._add_workspace_colliders(proto)
        particle_start = proto.particle_count
        self._add_mpm_pile(proto)
        particle_end = proto.particle_count

        arm_q_ids, arm_qd_ids = self._resolve_arm_joint_ids(proto)
        if len(arm_q_ids) != self.cfg.arm_dof_count:
            raise RuntimeError(f"Expected {self.cfg.arm_dof_count} UR10 arm DOFs, found {len(arm_q_ids)}.")

        return proto, {
            "body_count": proto.body_count,
            "joint_count": proto.joint_count,
            "shape_count": proto.shape_count,
            "particle_count": particle_end - particle_start,
            "particle_collider_shapes": [*paddle_shapes, *workspace_shapes],
            "arm_q_ids": arm_q_ids,
            "arm_qd_ids": arm_qd_ids,
            "ee_body": ee_body,
            "paddle_body": ee_body,
        }

    def _disable_robot_particle_collisions(self, builder: newton.ModelBuilder) -> None:
        """Keep MPM particle coupling focused on the pan, table, and bin."""
        particle_collision = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        for shape_id in range(builder.shape_count):
            builder.shape_flags[shape_id] &= ~particle_collision

    def _add_paddle_pad(self, builder: newton.ModelBuilder, body_id: int) -> list[int]:
        """Add one flat pad collider, like a tennis paddle face."""
        sx, sy, sz = self.cfg.paddle_size
        cfg = newton.ModelBuilder.ShapeConfig(
            mu=1.0,
            density=500.0,
            margin=self.cfg.paddle_collision_margin,
            gap=0.01,
        )

        return [
            builder.add_shape_box(
                body_id,
                xform=wp.transform(wp.vec3(*self.cfg.paddle_ee_offset), wp.quat_identity()),
                hx=0.5 * sx,
                hy=0.5 * sy,
                hz=0.5 * sz,
                cfg=cfg,
                color=(0.1, 0.25, 0.85),
            )
        ]

    def _configure_ur10_joints(self, builder: newton.ModelBuilder) -> None:
        initial_q = {
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        }
        arm_q_ids, arm_qd_ids = self._resolve_arm_joint_ids(builder)
        for q_id, qd_id, joint_name in zip(arm_q_ids, arm_qd_ids, self.cfg.arm_joint_names):
            builder.joint_q[q_id] = initial_q[joint_name]
            builder.joint_target_pos[qd_id] = initial_q[joint_name]
        for dof_id in range(builder.joint_dof_count):
            builder.joint_target_ke[dof_id] = 800.0
            builder.joint_target_kd[dof_id] = 60.0
            builder.joint_effort_limit[dof_id] = 100.0
            builder.joint_armature[dof_id] = 0.15
            builder.joint_target_mode[dof_id] = int(JointTargetMode.POSITION)

    def _resolve_arm_joint_ids(self, builder: newton.ModelBuilder) -> tuple[list[int], list[int]]:
        q_ids = []
        qd_ids = []
        for joint_name in self.cfg.arm_joint_names:
            matches = [i for i, label in enumerate(builder.joint_label) if label.endswith(joint_name)]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one joint matching {joint_name!r}, found {matches}.")
            joint_id = matches[0]
            q_ids.append(builder.joint_q_start[joint_id])
            qd_ids.append(builder.joint_qd_start[joint_id])
        return q_ids, qd_ids

    def _add_workspace_colliders(self, builder: newton.ModelBuilder) -> list[int]:
        workspace_shapes: list[int] = []
        workspace_shapes.extend(self._add_table(builder))
        workspace_shapes.extend(self._add_deep_side_bin(builder))
        return workspace_shapes

    def _add_table(self, builder: newton.ModelBuilder) -> list[int]:
        shapes: list[int] = []
        tx, ty, tz = self.cfg.table_center
        sx, sy, sz = self.cfg.table_size
        table_cfg = newton.ModelBuilder.ShapeConfig(mu=0.8, density=0.0, margin=0.01, gap=0.01)
        shapes.append(
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(tx, ty, tz), wp.quat_identity()),
                hx=0.5 * sx,
                hy=0.5 * sy,
                hz=0.5 * sz,
                cfg=table_cfg,
                color=(0.45, 0.34, 0.24),
            )
        )
        leg_sx, leg_sy, leg_sz = self.cfg.table_leg_size
        leg_z = tz - 0.5 * sz - 0.5 * leg_sz
        for leg_x in (tx - 0.43 * sx, tx + 0.43 * sx):
            for leg_y in (ty - 0.43 * sy, ty + 0.43 * sy):
                shapes.append(
                    builder.add_shape_box(
                        -1,
                        xform=wp.transform(wp.vec3(leg_x, leg_y, leg_z), wp.quat_identity()),
                        hx=0.5 * leg_sx,
                        hy=0.5 * leg_sy,
                        hz=0.5 * leg_sz,
                        cfg=table_cfg,
                        color=(0.32, 0.24, 0.16),
                    )
                )
        return shapes

    def _add_deep_side_bin(self, builder: newton.ModelBuilder) -> list[int]:
        workspace_shapes: list[int] = []
        wall_thickness = self.cfg.bin_wall_thickness
        bin_x, bin_y, _ = self.cfg.bin_center
        bin_half_x, bin_half_y, _ = self.cfg.bin_inner_half_extents
        bin_lower_z = self.cfg.bin_center[2] - self.cfg.bin_inner_half_extents[2]
        bin_wall_height = self.cfg.bin_wall_height
        wall_z = bin_lower_z + 0.5 * bin_wall_height
        front_wall_z = bin_lower_z + 0.5 * self.cfg.bin_front_wall_height
        bottom_z = bin_lower_z - 0.5 * wall_thickness
        wall_cfg = newton.ModelBuilder.ShapeConfig(mu=0.8, density=0.0, margin=0.01, gap=0.01)
        workspace_shapes.append(
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(bin_x, bin_y, bottom_z), wp.quat_identity()),
                hx=bin_half_x + 0.5 * wall_thickness,
                hy=bin_half_y + 0.5 * wall_thickness,
                hz=0.5 * wall_thickness,
                cfg=wall_cfg,
                color=(0.08, 0.14, 0.26),
            )
        )
        walls = [
            (
                (2.0 * bin_half_x + wall_thickness, wall_thickness, bin_wall_height),
                (bin_x, bin_y + bin_half_y + 0.5 * wall_thickness, wall_z),
            ),
            (
                (2.0 * bin_half_x + wall_thickness, wall_thickness, bin_wall_height),
                (bin_x, bin_y - bin_half_y - 0.5 * wall_thickness, wall_z),
            ),
            (
                (wall_thickness, 2.0 * bin_half_y + 2.0 * wall_thickness, bin_wall_height),
                (bin_x + bin_half_x + 0.5 * wall_thickness, bin_y, wall_z),
            ),
            (
                (wall_thickness, 2.0 * bin_half_y + 2.0 * wall_thickness, self.cfg.bin_front_wall_height),
                (bin_x - bin_half_x - 0.5 * wall_thickness, bin_y, front_wall_z),
            ),
        ]
        for size, pos in walls:
            workspace_shapes.append(
                builder.add_shape_box(
                    -1,
                    xform=wp.transform(wp.vec3(*pos), wp.quat_identity()),
                    hx=0.5 * size[0],
                    hy=0.5 * size[1],
                    hz=0.5 * size[2],
                    cfg=wall_cfg,
                    color=(0.1, 0.18, 0.32),
                )
            )
        rim_z = bin_lower_z + bin_wall_height + 0.5 * self.cfg.bin_rim_height
        rim_cfg = newton.ModelBuilder.ShapeConfig(mu=0.9, density=0.0, margin=0.01, gap=0.01)
        rim_segments = [
            (
                (2.0 * bin_half_x + 2.0 * wall_thickness, self.cfg.bin_rim_thickness, self.cfg.bin_rim_height),
                (bin_x, bin_y + bin_half_y + 0.5 * wall_thickness, rim_z),
            ),
            (
                (2.0 * bin_half_x + 2.0 * wall_thickness, self.cfg.bin_rim_thickness, self.cfg.bin_rim_height),
                (bin_x, bin_y - bin_half_y - 0.5 * wall_thickness, rim_z),
            ),
            (
                (self.cfg.bin_rim_thickness, 2.0 * bin_half_y + 2.0 * wall_thickness, self.cfg.bin_rim_height),
                (bin_x + bin_half_x + 0.5 * wall_thickness, bin_y, rim_z),
            ),
        ]
        for size, pos in rim_segments:
            workspace_shapes.append(
                builder.add_shape_box(
                    -1,
                    xform=wp.transform(wp.vec3(*pos), wp.quat_identity()),
                    hx=0.5 * size[0],
                    hy=0.5 * size[1],
                    hz=0.5 * size[2],
                    cfg=rim_cfg,
                    color=(0.12, 0.26, 0.48),
                )
            )
        return workspace_shapes

    def _add_mpm_pile(self, builder: newton.ModelBuilder) -> None:
        lo = np.array(self.cfg.pile_lo, dtype=np.float64)
        hi = np.array(self.cfg.pile_hi, dtype=np.float64)
        res = np.maximum(np.ceil(self.cfg.particles_per_cell * (hi - lo) / self.cfg.voxel_size), 1).astype(int)
        cell_size = (hi - lo) / res
        radius = float(np.max(cell_size) * 0.5)
        mass = float(np.prod(cell_size) * self.cfg.sand_density)
        builder.add_particle_grid(
            pos=wp.vec3(lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=int(res[0]) + 1,
            dim_y=int(res[1]) + 1,
            dim_z=int(res[2]) + 1,
            cell_x=float(cell_size[0]),
            cell_y=float(cell_size[1]),
            cell_z=float(cell_size[2]),
            mass=mass,
            jitter=2.0 * radius,
            radius_mean=radius,
            custom_attributes={
                "mpm:friction": self.cfg.sand_friction,
                "mpm:damping": self.cfg.sand_damping,
                "mpm:young_modulus": self.cfg.sand_young_modulus,
                "mpm:yield_pressure": self.cfg.sand_yield_pressure,
                "mpm:tensile_yield_ratio": self.cfg.sand_tensile_yield_ratio,
            },
        )

    @staticmethod
    def _find_body(builder: newton.ModelBuilder, body_name: str) -> int:
        matches = [i for i, label in enumerate(builder.body_label) if label.endswith(body_name)]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one body matching {body_name!r}, found {matches}.")
        return matches[0]

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._apply_viewer_forces()
        raw_actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        smoothing = float(self.cfg.action_smoothing_factor)
        self._actions = torch.lerp(self._actions, raw_actions, smoothing).clamp(-1.0, 1.0)
        joint_q = wp.to_torch(NewtonManager.get_state_0().joint_q)[self._joint_q_ids]
        self._target_paddle_pos = self._replace_nonfinite(self._target_paddle_pos, self._previous_paddle_pos)
        self._target_paddle_pos = self._clamp_paddle_target_position(
            self._target_paddle_pos + self._actions[:, :3] * self.cfg.cartesian_position_action_scale * self.step_dt
        )
        target_ee_quat = self._target_paddle_quat()
        self._joint_targets = self._solve_newton_ik(
            joint_q,
            self._target_paddle_pos,
            target_ee_quat,
            self.cfg.ik_action_iterations,
            self.cfg.max_ik_delta_q,
        )

    def _apply_action(self) -> None:
        control = NewtonManager.get_control()
        wp.to_torch(control.joint_target_pos)[self._joint_qd_ids] = self._joint_targets

    def _solve_newton_ik(
        self,
        current_joint_q: torch.Tensor,
        target_paddle_pos: torch.Tensor,
        target_ee_quat: torch.Tensor,
        iterations: int,
        max_delta_q: float,
    ) -> torch.Tensor:
        self._ik_joint_q_in_t[:, :] = self._ik_default_joint_q.unsqueeze(0)
        self._ik_joint_q_in_t[:, self._ik_arm_q_ids] = current_joint_q.to(self._ik_joint_q_in_t.device)
        self._ik_target_positions_t[:, :] = target_paddle_pos.to(self._ik_target_positions_t.device)
        self._ik_target_rotations_t[:, :] = self._normalize_quat(target_ee_quat).to(self._ik_target_rotations_t.device)

        self._ik_solver.step(
            self._ik_joint_q_in,
            self._ik_joint_q_out,
            iterations=iterations,
            step_size=self.cfg.ik_step_size,
        )
        solved_joint_q = self._ik_joint_q_out_t[:, self._ik_arm_q_ids].to(current_joint_q.device)
        joint_delta = torch.nan_to_num(
            solved_joint_q - current_joint_q,
            nan=0.0,
            posinf=max_delta_q,
            neginf=-max_delta_q,
        )
        joint_delta = torch.clamp(joint_delta, -max_delta_q, max_delta_q)
        return torch.clamp(current_joint_q + joint_delta, self._joint_lower, self._joint_upper)

    def _clamp_paddle_target_position(self, position: torch.Tensor) -> torch.Tensor:
        return torch.clamp(position, min=self._paddle_workspace_lower, max=self._paddle_workspace_upper)

    def _get_observations(self) -> dict:
        particle_pos = self._sanitize_particle_pos(self._particle_pos_e())
        heightmap_grid, density_grid = self._particle_grid_observations(particle_pos)
        gridmap = torch.stack((heightmap_grid, density_grid), dim=1)
        state = NewtonManager.get_state_0()
        joint_q = self._replace_nonfinite(wp.to_torch(state.joint_q)[self._joint_q_ids], self._default_joint_q)
        joint_qd = self._replace_nonfinite(wp.to_torch(state.joint_qd)[self._joint_qd_ids], 0.0)
        bin_fraction = self._count_particles_in_bin(particle_pos)[:, None] / float(self._particle_count)
        paddle_corners = self._paddle_corners_e()
        particle_centroid = self._scoopable_particle_centroid(particle_pos)
        particle_centroid_velocity = (particle_centroid - self._previous_particle_centroid) / self.step_dt
        particle_centroid_velocity = torch.clamp(torch.nan_to_num(particle_centroid_velocity), -2.0, 2.0)
        self._last_particle_centroid_velocity = particle_centroid_velocity
        self._previous_particle_centroid = particle_centroid
        paddle_pos = self._paddle_pos_e()
        paddle_velocity = self._last_paddle_velocity
        bin_target = self._bin_target.unsqueeze(0).expand(self.num_envs, -1)
        bin_mouth = self._bin_mouth_center.unsqueeze(0).expand(self.num_envs, -1)
        vector_to_mouth = self._normalize_vector(bin_mouth - paddle_pos)
        vector_paddle_to_centroid = self._normalize_vector(particle_centroid - paddle_pos)
        vector_centroid_to_mouth = self._normalize_vector(bin_mouth - particle_centroid)
        push_direction = self._push_direction_to_bin()
        paddle_normal = self._paddle_normal_e()
        contact_score = self._paddle_particle_contact_score(particle_pos)
        setup_features = self._paddle_push_setup_features(paddle_pos, particle_centroid, contact_score)
        paddle_push_speed = torch.sum(paddle_velocity * push_direction, dim=-1, keepdim=True)
        particle_push_speed = torch.sum(particle_centroid_velocity * push_direction, dim=-1, keepdim=True)
        spill_fraction = self._particles_spilled(particle_pos).float().mean(dim=1, keepdim=True)
        particle_progress = self._particle_progress_toward_bin(particle_pos)[:, None]
        bin_proximity = self._particle_bin_proximity(particle_pos)[:, None]
        proprio = torch.cat(
            (
                self._normalize_joint_positions(joint_q),
                self._normalize_joint_velocities(joint_qd),
                self._normalize_positions(paddle_pos),
                self._normalize_positions(paddle_corners).reshape(self.num_envs, -1),
                self._normalize_positions(particle_centroid),
                0.25 * particle_centroid_velocity,
                0.2 * paddle_velocity,
                self._normalize_positions(bin_target),
                self._normalize_positions(bin_mouth),
                vector_to_mouth,
                vector_paddle_to_centroid,
                vector_centroid_to_mouth,
                push_direction,
                paddle_normal,
                setup_features,
                torch.clamp(paddle_push_speed / max(float(self.cfg.paddle_push_speed_norm), 1.0e-6), -1.0, 1.0),
                torch.clamp(particle_push_speed / max(float(self.cfg.particle_push_speed_norm), 1.0e-6), -1.0, 1.0),
                bin_fraction,
                spill_fraction,
                particle_progress,
                bin_proximity,
                self._previous_actions,
            ),
            dim=-1,
        )
        nonfinite_proprio = ~torch.isfinite(proprio).all(dim=1)
        if torch.any(nonfinite_proprio):
            self._last_nonfinite_observation = nonfinite_proprio
            self.extras.setdefault("log", {})["Diagnostics/nonfinite_proprio"] = nonfinite_proprio.float().mean().item()
            proprio = torch.nan_to_num(proprio, nan=0.0, posinf=5.0, neginf=-5.0)
        self._policy_observation_spheres.update(
            self, heightmap_grid, density_grid, paddle_corners, particle_centroid, particle_pos
        )
        privileged = self._privileged_observations(particle_pos, paddle_velocity)
        return {
            "gridmap": torch.nan_to_num(torch.clamp(gridmap, 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0),
            "proprio": torch.nan_to_num(torch.clamp(proprio, -2.0, 2.0), nan=0.0, posinf=2.0, neginf=-2.0),
            "privileged": torch.nan_to_num(torch.clamp(privileged, -2.0, 2.0), nan=0.0, posinf=2.0, neginf=-2.0),
        }

    def _get_rewards(self) -> torch.Tensor:
        particle_pos = self._sanitize_particle_pos(self._particle_pos_e())
        count = self._count_particles_in_bin(particle_pos)
        bin_fraction = count / float(self._particle_count)
        delta_bin_fraction = bin_fraction - self._previous_bin_fraction
        self._previous_bin_count = count
        self._previous_bin_fraction = bin_fraction
        progress = self._particle_progress_toward_bin(particle_pos)
        delta_progress = progress - self._previous_particle_progress
        self._previous_particle_progress = progress
        bin_proximity = self._particle_bin_proximity(particle_pos)
        self._previous_bin_proximity = bin_proximity
        spill_fraction = self._particles_spilled(particle_pos).float().mean(dim=1)
        mouth_entry = self._particle_mouth_entry(particle_pos)
        particle_centroid = self._scoopable_particle_centroid(particle_pos)
        particle_centroid_velocity = (particle_centroid - self._previous_particle_centroid) / self.step_dt
        particle_centroid_velocity = torch.nan_to_num(particle_centroid_velocity, nan=0.0, posinf=2.0, neginf=-2.0)
        particle_centroid_velocity = torch.clamp(particle_centroid_velocity, -2.0, 2.0)
        centroid_progress = self._particle_centroid_progress_toward_bin(particle_centroid)
        delta_centroid_progress = centroid_progress - self._previous_centroid_progress
        self._previous_centroid_progress = centroid_progress

        paddle_pos = self._paddle_pos_e()
        paddle_corners = self._paddle_corners_e()
        paddle_distance = torch.linalg.norm(paddle_pos - self._current_pile_center, dim=-1)
        paddle_proximity = torch.exp(-4.0 * paddle_distance)
        paddle_bin_proximity = self._paddle_bin_proximity(paddle_pos)
        paddle_orientation = self._paddle_push_orientation()
        paddle_contact = self._paddle_particle_contact_score(particle_pos)
        setup_features = self._paddle_push_setup_features(paddle_pos, particle_centroid, paddle_contact)
        paddle_setup = setup_features[:, 0]
        paddle_low_penalty = torch.square(
            torch.clamp((self.cfg.paddle_min_height - paddle_corners[..., 2].amin(dim=1)) / 0.10, min=0.0)
        )
        paddle_low_penalty = torch.clamp(paddle_low_penalty, max=1.0)
        paddle_velocity = (paddle_pos - self._previous_paddle_pos) / self.step_dt
        paddle_velocity = torch.nan_to_num(
            paddle_velocity,
            nan=0.0,
            posinf=self.cfg.max_paddle_speed,
            neginf=-self.cfg.max_paddle_speed,
        )
        paddle_velocity = torch.clamp(paddle_velocity, -self.cfg.max_paddle_speed, self.cfg.max_paddle_speed)
        paddle_speed = torch.linalg.norm(paddle_velocity, dim=-1)
        paddle_speed = torch.nan_to_num(paddle_speed, nan=self.cfg.max_paddle_speed, posinf=self.cfg.max_paddle_speed)
        paddle_speed = torch.clamp(paddle_speed, max=self.cfg.max_paddle_speed)
        push_direction = self._push_direction_to_bin()
        paddle_forward_velocity = torch.sum(paddle_velocity * push_direction, dim=-1)
        particle_forward_velocity = torch.sum(particle_centroid_velocity * push_direction, dim=-1)
        paddle_push_speed = torch.clamp(
            paddle_forward_velocity / max(float(self.cfg.paddle_push_speed_norm), 1.0e-6), min=0.0, max=1.0
        )
        particle_push_speed = torch.clamp(
            particle_forward_velocity / max(float(self.cfg.particle_push_speed_norm), 1.0e-6), min=0.0, max=1.0
        )
        paddle_retreat_speed = torch.clamp(
            -paddle_forward_velocity / max(float(self.cfg.paddle_push_speed_norm), 1.0e-6), min=0.0, max=1.0
        )
        push_gate = torch.clamp(0.5 * paddle_setup + 0.5 * paddle_contact, 0.0, 1.0)
        self._previous_paddle_pos = paddle_pos
        self._last_paddle_velocity = paddle_velocity
        joint_qd = wp.to_torch(NewtonManager.get_state_0().joint_qd)[self._joint_qd_ids]
        joint_qd = torch.nan_to_num(
            joint_qd,
            nan=0.0,
            posinf=self.cfg.max_joint_velocity,
            neginf=-self.cfg.max_joint_velocity,
        )
        joint_qd = torch.clamp(joint_qd, -self.cfg.max_joint_velocity, self.cfg.max_joint_velocity)
        action_penalty = torch.sum(torch.square(self._actions), dim=-1)
        action_delta = self._actions - self._previous_actions
        action_rate_penalty = torch.sum(torch.square(action_delta), dim=-1)
        joint_velocity_penalty = torch.sum(torch.square(joint_qd), dim=-1)
        success = bin_fraction >= self._target_success_fraction()
        nonfinite_failure = self._nonfinite_state.float()
        self._episode_succeeded |= success
        self._previous_actions = self._actions.clone()

        rewards = {
            "bin_fraction": self.cfg.reward_bin_fraction_scale * bin_fraction,
            "delta_bin_fraction": self.cfg.reward_delta_bin_fraction_scale * torch.clamp(delta_bin_fraction, min=0.0),
            "particle_progress": self.cfg.reward_particle_progress_scale * torch.clamp(delta_progress, min=0.0),
            "centroid_progress": self.cfg.reward_centroid_progress_scale
            * torch.clamp(delta_centroid_progress, min=0.0),
            "bin_proximity": self.cfg.reward_bin_proximity_scale * bin_proximity,
            "mouth_entry": self.cfg.reward_mouth_entry_scale * mouth_entry,
            "spill_penalty": -self.cfg.reward_spill_penalty_scale * spill_fraction,
            "paddle_proximity": self.cfg.reward_paddle_proximity_scale * paddle_proximity,
            "paddle_bin_proximity": self.cfg.reward_paddle_bin_proximity_scale * paddle_bin_proximity,
            "paddle_orientation": self.cfg.reward_paddle_orientation_scale * paddle_orientation,
            "paddle_setup": self.cfg.reward_paddle_setup_scale * paddle_setup,
            "paddle_contact": self.cfg.reward_paddle_contact_scale * paddle_contact,
            "paddle_push_velocity": self.cfg.reward_paddle_push_velocity_scale * paddle_push_speed * push_gate,
            "particle_push_velocity": self.cfg.reward_particle_push_velocity_scale
            * particle_push_speed
            * paddle_contact,
            "paddle_retreat_penalty": -self.cfg.reward_paddle_retreat_penalty_scale * paddle_retreat_speed * push_gate,
            "paddle_low_penalty": -self.cfg.reward_paddle_low_penalty_scale * paddle_low_penalty,
            "paddle_speed_penalty": -self.cfg.reward_paddle_speed_penalty_scale * torch.square(paddle_speed),
            "action_penalty": -self.cfg.action_penalty_scale * action_penalty,
            "action_rate_penalty": -self.cfg.action_rate_penalty_scale * action_rate_penalty,
            "joint_velocity_penalty": -self.cfg.joint_velocity_penalty_scale * joint_velocity_penalty,
            "success_bonus": self.cfg.reward_success_bonus * success.float(),
            "nonfinite_penalty": -self.cfg.reward_nonfinite_penalty_scale * nonfinite_failure,
        }
        rewards = {name: torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0) for name, value in rewards.items()}
        reward = torch.nan_to_num(
            torch.stack(list(rewards.values()), dim=0).sum(dim=0), nan=0.0, posinf=0.0, neginf=0.0
        )
        for name, value in rewards.items():
            self._episode_sums[name] += value
        self._last_bin_fraction = bin_fraction
        self._last_spill_fraction = spill_fraction
        self._last_particle_progress = progress
        self._last_centroid_progress = centroid_progress
        self._last_bin_proximity = bin_proximity
        self._last_mouth_entry = mouth_entry
        self._last_paddle_speed = paddle_speed
        self._last_action_rate = torch.linalg.norm(action_delta, dim=-1)
        self._last_paddle_bin_proximity = paddle_bin_proximity
        self._last_paddle_orientation = paddle_orientation
        self._last_paddle_setup = paddle_setup
        self._last_paddle_contact = paddle_contact
        self._last_paddle_push_speed = paddle_push_speed
        self._last_particle_push_speed = particle_push_speed
        self._last_paddle_retreat_speed = paddle_retreat_speed
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        particle_pos_raw = self._particle_pos_e()
        particle_pos = self._sanitize_particle_pos(particle_pos_raw)
        joint_q = wp.to_torch(NewtonManager.get_state_0().joint_q)[self._joint_q_ids]
        joint_qd = wp.to_torch(NewtonManager.get_state_0().joint_qd)[self._joint_qd_ids]
        ee_pose = wp.to_torch(NewtonManager.get_state_0().body_q)[self._ee_body_ids]
        self._nonfinite_particles = ~torch.isfinite(particle_pos_raw).flatten(start_dim=1).all(dim=1)
        self._nonfinite_joint_q = ~torch.isfinite(joint_q).all(dim=1)
        self._nonfinite_joint_qd = ~torch.isfinite(joint_qd).all(dim=1)
        self._nonfinite_body = ~torch.isfinite(ee_pose).all(dim=1)
        self._nonfinite_state = (
            self._nonfinite_particles | self._nonfinite_joint_q | self._nonfinite_joint_qd | self._nonfinite_body
        )
        bin_fraction = self._count_particles_in_bin(particle_pos) / float(self._particle_count)
        task_success = bin_fraction >= self._target_success_fraction()
        terminated = task_success | self._nonfinite_state
        self._episode_succeeded |= task_success
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        env_ids = env_ids.long()

        log_extras = None
        if hasattr(self, "_episode_sums"):
            completed_success_rate = self._episode_succeeded[env_ids].float().mean().item()
            completed_nonfinite_rate = self._nonfinite_state[env_ids].float().mean().item()
            completed_bin_fraction = self._last_bin_fraction[env_ids].mean().item()
            self._update_curriculum(
                completed_success_rate,
                completed_nonfinite_rate,
                completed_bin_fraction,
                int(env_ids.numel()),
            )
            extras = {}
            for key, value in self._episode_sums.items():
                extras[f"Episode_Reward/{key}"] = value[env_ids].mean().item() / self.max_episode_length_s
                value[env_ids] = 0.0
            extras["Metrics/particles_in_bin"] = self._previous_bin_count[env_ids].mean().item()
            extras["Metrics/particle_fraction_in_bin"] = self._last_bin_fraction[env_ids].mean().item()
            extras["Metrics/spill_fraction"] = self._last_spill_fraction[env_ids].mean().item()
            extras["Metrics/particle_progress"] = self._last_particle_progress[env_ids].mean().item()
            extras["Metrics/centroid_progress"] = self._last_centroid_progress[env_ids].mean().item()
            extras["Metrics/bin_proximity"] = self._last_bin_proximity[env_ids].mean().item()
            extras["Metrics/mouth_entry"] = self._last_mouth_entry[env_ids].mean().item()
            extras["Metrics/paddle_speed"] = self._last_paddle_speed[env_ids].mean().item()
            extras["Metrics/action_rate"] = self._last_action_rate[env_ids].mean().item()
            extras["Metrics/paddle_bin_proximity"] = self._last_paddle_bin_proximity[env_ids].mean().item()
            extras["Metrics/paddle_orientation"] = self._last_paddle_orientation[env_ids].mean().item()
            extras["Metrics/paddle_setup"] = self._last_paddle_setup[env_ids].mean().item()
            extras["Metrics/paddle_contact"] = self._last_paddle_contact[env_ids].mean().item()
            extras["Metrics/paddle_push_speed"] = self._last_paddle_push_speed[env_ids].mean().item()
            extras["Metrics/particle_push_speed"] = self._last_particle_push_speed[env_ids].mean().item()
            extras["Metrics/paddle_retreat_speed"] = self._last_paddle_retreat_speed[env_ids].mean().item()
            extras["Metrics/success_rate"] = self._episode_succeeded[env_ids].float().mean().item()
            extras["Diagnostics/nonfinite_state_rate"] = self._nonfinite_state[env_ids].float().mean().item()
            extras["Diagnostics/nonfinite_particles_rate"] = self._nonfinite_particles[env_ids].float().mean().item()
            extras["Diagnostics/nonfinite_joint_q_rate"] = self._nonfinite_joint_q[env_ids].float().mean().item()
            extras["Diagnostics/nonfinite_joint_qd_rate"] = self._nonfinite_joint_qd[env_ids].float().mean().item()
            extras["Diagnostics/nonfinite_body_rate"] = self._nonfinite_body[env_ids].float().mean().item()
            extras["Diagnostics/nonfinite_observation_rate"] = (
                self._last_nonfinite_observation[env_ids].float().mean().item()
            )
            extras["Diagnostics/reset_ik_failure_rate"] = self._last_reset_ik_failure[env_ids].float().mean().item()
            extras["Diagnostics/reset_ik_position_error"] = self._last_reset_ik_position_error[env_ids].mean().item()
            extras["Curriculum/stage"] = float(self._curriculum_stage)
            extras["Curriculum/success_ema"] = float(self._curriculum_success_ema)
            extras["Curriculum/bin_fraction_ema"] = float(self._curriculum_bin_fraction_ema)
            extras["Curriculum/target_success_fraction"] = float(self._target_success_fraction())
            self._episode_succeeded[env_ids] = False
            log_extras = extras

        super()._reset_idx(env_ids)
        state_0 = NewtonManager.get_state_0()
        state_1 = NewtonManager.get_state_1()
        control = NewtonManager.get_control()
        joint_q = wp.to_torch(state_0.joint_q)
        joint_qd = wp.to_torch(state_0.joint_qd)
        particle_q = wp.to_torch(state_0.particle_q)
        particle_qd = wp.to_torch(state_0.particle_qd)

        joint_q[self._joint_q_ids[env_ids]] = self._default_joint_q[env_ids]
        joint_qd[self._joint_qd_ids[env_ids]] = self._default_joint_qd[env_ids]
        reset_particle_q = self._sample_particle_reset_positions(env_ids)
        if log_extras is not None:
            log_extras["Curriculum/reset_pile_center_x"] = self._current_pile_center[env_ids, 0].mean().item()
            log_extras["Curriculum/reset_pile_center_y"] = self._current_pile_center[env_ids, 1].mean().item()
            log_extras["Curriculum/reset_pile_scale_x"] = self._current_pile_half_extents[
                env_ids, 0
            ].mean().item() / max(float(self._default_pile_half_extents[0]), 1.0e-6)
            self.extras["log"] = log_extras
        particle_q[self._particle_ids[env_ids]] = reset_particle_q
        particle_qd[self._particle_ids[env_ids]] = 0.0
        wp.to_torch(state_1.joint_q)[self._joint_q_ids[env_ids]] = self._default_joint_q[env_ids]
        wp.to_torch(state_1.joint_qd)[self._joint_qd_ids[env_ids]] = self._default_joint_qd[env_ids]
        wp.to_torch(state_1.particle_q)[self._particle_ids[env_ids]] = reset_particle_q
        wp.to_torch(state_1.particle_qd)[self._particle_ids[env_ids]] = 0.0

        self._reset_robot_to_curriculum_pose(env_ids, state_0, state_1, control)
        if log_extras is not None:
            reset_paddle = self._paddle_pos_e()[env_ids]
            log_extras["Curriculum/reset_paddle_x"] = reset_paddle[:, 0].mean().item()
            log_extras["Curriculum/reset_paddle_y"] = reset_paddle[:, 1].mean().item()
            log_extras["Curriculum/reset_paddle_z"] = reset_paddle[:, 2].mean().item()

        self._joint_targets[env_ids] = self._default_joint_q[env_ids]
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._joint_targets[env_ids] = wp.to_torch(state_0.joint_q)[self._joint_q_ids[env_ids]]
        wp.to_torch(control.joint_target_pos)[self._joint_qd_ids[env_ids]] = self._joint_targets[env_ids]
        if hasattr(self, "_mpm_solver"):
            self._mpm_solver._last_step_data.save_collider_current_position(state_0.body_q)
        particle_pos = self._particle_pos_e()
        self._previous_bin_count[env_ids] = self._count_particles_in_bin(particle_pos)[env_ids]
        self._previous_bin_fraction[env_ids] = self._previous_bin_count[env_ids] / float(self._particle_count)
        if log_extras is not None:
            log_extras["Curriculum/reset_particles_in_bin"] = self._previous_bin_count[env_ids].mean().item()
            log_extras["Curriculum/reset_bin_fraction"] = self._previous_bin_fraction[env_ids].mean().item()
        self._previous_particle_progress[env_ids] = self._particle_progress_toward_bin(particle_pos)[env_ids]
        particle_centroid = self._scoopable_particle_centroid(particle_pos)
        self._previous_centroid_progress[env_ids] = self._particle_centroid_progress_toward_bin(particle_centroid)[
            env_ids
        ]
        self._previous_bin_proximity[env_ids] = self._particle_bin_proximity(particle_pos)[env_ids]
        self._previous_paddle_pos[env_ids] = self._paddle_pos_e()[env_ids]
        self._target_paddle_pos[env_ids] = self._previous_paddle_pos[env_ids]
        self._previous_particle_centroid[env_ids] = particle_centroid[env_ids]
        self._last_centroid_progress[env_ids] = self._previous_centroid_progress[env_ids]
        self._last_paddle_setup[env_ids] = 0.0
        self._last_paddle_contact[env_ids] = self._paddle_particle_contact_score(particle_pos)[env_ids]
        self._last_paddle_push_speed[env_ids] = 0.0
        self._last_particle_push_speed[env_ids] = 0.0
        self._last_paddle_retreat_speed[env_ids] = 0.0
        self._last_paddle_velocity[env_ids] = 0.0
        self._last_particle_centroid_velocity[env_ids] = 0.0
        self._nonfinite_state[env_ids] = False
        self._nonfinite_particles[env_ids] = False
        self._nonfinite_joint_q[env_ids] = False
        self._nonfinite_joint_qd[env_ids] = False
        self._nonfinite_body[env_ids] = False
        self._last_nonfinite_observation[env_ids] = False
        self._last_reset_ik_failure[env_ids] = False
        self._last_reset_ik_position_error[env_ids] = 0.0

    def _particle_grid_observations(self, particle_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        particle_pos = self._sanitize_particle_pos(particle_pos)
        map_size = self.cfg.heightmap_size
        rel_x = (particle_pos[..., 0] - self._heightmap_x_min) / self._heightmap_x_range
        rel_y = (particle_pos[..., 1] - self._heightmap_y_min) / self._heightmap_y_range
        px = torch.clamp((rel_x * map_size).long(), 0, map_size - 1)
        py = torch.clamp((rel_y * map_size).long(), 0, map_size - 1)
        particle_height = torch.clamp(
            (particle_pos[..., 2] - self._heightmap_z_min) / self.cfg.heightmap_z_range, 0.0, 1.0
        )
        valid = (
            torch.isfinite(particle_pos).all(dim=-1) & (rel_x >= 0.0) & (rel_x < 1.0) & (rel_y >= 0.0) & (rel_y < 1.0)
        )
        particle_height = torch.where(
            valid,
            torch.clamp(particle_height, min=self.cfg.heightmap_occupied_cell_value),
            particle_height,
        )
        flat_indices = self._heightmap_env_offsets + py * map_size + px
        flat_values = torch.where(valid, particle_height, torch.zeros_like(particle_height))
        height = torch.zeros(self.num_envs * map_size * map_size, device=self.device)
        height.scatter_reduce_(0, flat_indices.reshape(-1), flat_values.reshape(-1), reduce="amax", include_self=True)
        density_values = torch.where(valid, torch.ones_like(particle_height), torch.zeros_like(particle_height))
        density = torch.zeros_like(height)
        density.scatter_add_(0, flat_indices.reshape(-1), density_values.reshape(-1))
        density = torch.clamp(density / self.cfg.heightmap_density_norm, 0.0, 1.0)
        return height.reshape(self.num_envs, map_size, map_size), density.reshape(self.num_envs, map_size, map_size)

    def _particle_pos_e(self) -> torch.Tensor:
        particle_pos_w = wp.to_torch(NewtonManager.get_state_0().particle_q)[self._particle_ids]
        return particle_pos_w - self.scene.env_origins[:, None, :]

    def _sanitize_particle_pos(self, particle_pos: torch.Tensor) -> torch.Tensor:
        finite_particle = torch.isfinite(particle_pos).all(dim=-1, keepdim=True)
        if hasattr(self, "_nonfinite_state"):
            self._nonfinite_state |= ~finite_particle.squeeze(-1).all(dim=1)
        fallback = self._current_pile_center[:, None, :].expand_as(particle_pos)
        return torch.where(finite_particle, particle_pos, fallback)

    def _paddle_pose_e(self) -> tuple[torch.Tensor, torch.Tensor]:
        body_q = wp.to_torch(NewtonManager.get_state_0().body_q)
        ee_pose = body_q[self._ee_body_ids]
        if hasattr(self, "_nonfinite_state"):
            self._nonfinite_state |= ~torch.isfinite(ee_pose).all(dim=1)
        ee_pos = self._replace_nonfinite(ee_pose[:, :3], self.scene.env_origins)
        ee_quat = self._sanitize_quat(ee_pose[:, 3:7])
        paddle_pos = ee_pos + self._quat_rotate(ee_quat, self._paddle_center_offset)
        return paddle_pos - self.scene.env_origins, ee_quat

    def _paddle_pos_e(self) -> torch.Tensor:
        return self._paddle_pose_e()[0]

    def _paddle_corners_e(self) -> torch.Tensor:
        paddle_pos, paddle_quat = self._paddle_pose_e()
        corner_offsets = self._paddle_corner_offsets.unsqueeze(0).expand(self.num_envs, -1, -1)
        corner_quats = paddle_quat[:, None, :].expand(-1, corner_offsets.shape[1], -1)
        local_corner_offsets = corner_offsets - self._paddle_center_offset
        return paddle_pos[:, None, :] + self._quat_rotate(corner_quats, local_corner_offsets)

    def _paddle_normal_e(self) -> torch.Tensor:
        _, paddle_quat = self._paddle_pose_e()
        local_normal = torch.tensor((0.0, 0.0, 1.0), device=self.device).expand(self.num_envs, -1)
        return self._quat_rotate(paddle_quat, local_normal)

    def _push_direction_to_bin(self) -> torch.Tensor:
        push_direction = self._bin_mouth_center.unsqueeze(0) - self._current_pile_center
        push_direction = push_direction.clone()
        push_direction[:, 2] = 0.0
        return push_direction / torch.clamp(torch.linalg.norm(push_direction, dim=-1, keepdim=True), min=1.0e-6)

    def _target_paddle_quat(self) -> torch.Tensor:
        """Return a vertical blade orientation whose face normal points toward the bin."""
        push_direction = self._push_direction_to_bin()
        yaw = torch.atan2(push_direction[:, 1], push_direction[:, 0])
        half_yaw = 0.5 * yaw
        yaw_quat = torch.stack(
            (
                torch.zeros_like(half_yaw),
                torch.zeros_like(half_yaw),
                torch.sin(half_yaw),
                torch.cos(half_yaw),
            ),
            dim=-1,
        )
        pitch_half_angle = torch.full_like(half_yaw, 0.25 * torch.pi)
        pitch_quat = torch.stack(
            (
                torch.zeros_like(pitch_half_angle),
                torch.sin(pitch_half_angle),
                torch.zeros_like(pitch_half_angle),
                torch.cos(pitch_half_angle),
            ),
            dim=-1,
        )
        return self._normalize_quat(self._quat_multiply(yaw_quat, pitch_quat))

    def _paddle_bin_proximity(self, paddle_pos: torch.Tensor) -> torch.Tensor:
        normalized_error = self._scale_vector(self._bin_mouth_center.unsqueeze(0) - paddle_pos)
        distance = torch.linalg.norm(normalized_error, dim=-1)
        return torch.exp(-2.0 * torch.clamp(distance, max=4.0))

    def _paddle_push_orientation(self) -> torch.Tensor:
        normal = self._paddle_normal_e()
        push_direction = self._push_direction_to_bin()
        facing_bin = torch.clamp(torch.sum(normal * push_direction, dim=-1), min=0.0, max=1.0)
        vertical_blade = torch.exp(-12.0 * torch.square(normal[:, 2]))
        return torch.square(facing_bin) * vertical_blade

    def _paddle_particle_contact_score(self, particle_pos: torch.Tensor) -> torch.Tensor:
        paddle_pos, paddle_quat = self._paddle_pose_e()
        particle_delta = particle_pos - paddle_pos[:, None, :]
        local_delta = self._quat_rotate(self._quat_conjugate(paddle_quat)[:, None, :], particle_delta)
        margin = float(self.cfg.paddle_contact_margin)
        in_face_x = torch.abs(local_delta[..., 0]) <= self._paddle_half_size[0] + margin
        in_face_y = torch.abs(local_delta[..., 1]) <= self._paddle_half_size[1] + margin
        in_front = local_delta[..., 2] >= self._paddle_half_size[2] - margin
        in_reach = local_delta[..., 2] <= self._paddle_half_size[2] + self.cfg.paddle_contact_depth
        contact_count = (in_face_x & in_face_y & in_front & in_reach & self._particles_in_workspace(particle_pos)).sum(
            dim=1,
            dtype=torch.float32,
        )
        return 1.0 - torch.exp(-contact_count / max(float(self.cfg.paddle_contact_count_norm), 1.0e-6))

    def _paddle_push_setup_features(
        self, paddle_pos: torch.Tensor, particle_centroid: torch.Tensor, contact_score: torch.Tensor
    ) -> torch.Tensor:
        push_direction = self._push_direction_to_bin()
        rel_paddle = paddle_pos - particle_centroid
        longitudinal_offset = -torch.sum(rel_paddle * push_direction, dim=-1)
        target_offset = self._current_pile_half_extents[:, 0] + self.cfg.paddle_setup_distance
        behind_error = (longitudinal_offset - target_offset) / max(float(self.cfg.paddle_setup_distance_std), 1.0e-6)
        behind_score = torch.exp(-torch.square(behind_error))

        lateral_vector = rel_paddle - torch.sum(rel_paddle * push_direction, dim=-1, keepdim=True) * push_direction
        lateral_vector = lateral_vector.clone()
        lateral_vector[:, 2] = 0.0
        lateral_error = torch.linalg.norm(lateral_vector, dim=-1)
        lateral_score = torch.exp(-torch.square(lateral_error / max(float(self.cfg.paddle_setup_lateral_std), 1.0e-6)))

        target_height = self.cfg.table_top_z + self.cfg.paddle_setup_height_offset
        height_error = (paddle_pos[:, 2] - target_height) / max(float(self.cfg.paddle_setup_height_std), 1.0e-6)
        height_score = torch.exp(-torch.square(height_error))
        orientation_score = self._paddle_push_orientation()
        setup_score = behind_score * lateral_score * height_score * orientation_score
        setup_score = torch.maximum(setup_score, 0.25 * contact_score * orientation_score)
        return torch.stack((setup_score, contact_score, behind_score, lateral_score, height_score), dim=-1)

    def _normalize_positions(self, positions: torch.Tensor) -> torch.Tensor:
        norm_x = 2.0 * (positions[..., 0] - self._heightmap_x_min) / self._heightmap_x_range - 1.0
        norm_y = 2.0 * (positions[..., 1] - self._heightmap_y_min) / self._heightmap_y_range - 1.0
        norm_z = 2.0 * (positions[..., 2] - self._heightmap_z_min) / self.cfg.heightmap_z_range - 1.0
        return torch.stack((norm_x, norm_y, norm_z), dim=-1)

    def _normalize_joint_positions(self, joint_q: torch.Tensor) -> torch.Tensor:
        return torch.clamp((joint_q - self._joint_center) / self._joint_half_range, -1.0, 1.0)

    def _normalize_joint_velocities(self, joint_qd: torch.Tensor) -> torch.Tensor:
        joint_qd = torch.nan_to_num(
            joint_qd,
            nan=0.0,
            posinf=self.cfg.max_joint_velocity,
            neginf=-self.cfg.max_joint_velocity,
        )
        return torch.clamp(joint_qd / max(float(self.cfg.max_joint_velocity), 1.0e-6), -1.0, 1.0)

    def _scale_vector(self, vector: torch.Tensor) -> torch.Tensor:
        scale = torch.tensor(
            (self._heightmap_x_range, self._heightmap_y_range, self.cfg.heightmap_z_range), device=self.device
        )
        return vector / torch.clamp(scale, min=1.0e-6)

    def _normalize_vector(self, vector: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self._scale_vector(vector), -1.0, 1.0)

    @staticmethod
    def _quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        quat_xyz, vec = torch.broadcast_tensors(quat[..., :3], vec)
        quat_w = torch.broadcast_to(quat[..., 3:4], quat_xyz.shape[:-1] + (1,))
        cross = torch.cross(quat_xyz, vec, dim=-1)
        return vec + 2.0 * (quat_w * cross + torch.cross(quat_xyz, cross, dim=-1))

    @staticmethod
    def _quat_multiply(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        lhs_xyz, lhs_w = lhs[..., :3], lhs[..., 3:4]
        rhs_xyz, rhs_w = rhs[..., :3], rhs[..., 3:4]
        xyz = lhs_w * rhs_xyz + rhs_w * lhs_xyz + torch.cross(lhs_xyz, rhs_xyz, dim=-1)
        w = lhs_w * rhs_w - torch.sum(lhs_xyz * rhs_xyz, dim=-1, keepdim=True)
        return torch.cat((xyz, w), dim=-1)

    @staticmethod
    def _normalize_quat(quat: torch.Tensor) -> torch.Tensor:
        return quat / torch.clamp(torch.linalg.norm(quat, dim=-1, keepdim=True), min=1.0e-8)

    @staticmethod
    def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
        return torch.cat((-quat[..., :3], quat[..., 3:4]), dim=-1)

    def _robust_particle_centroid(self, particle_pos: torch.Tensor) -> torch.Tensor:
        particle_pos = self._sanitize_particle_pos(particle_pos)
        in_workspace = self._particles_in_workspace(particle_pos)
        weights = in_workspace.float()
        weighted_sum = torch.sum(particle_pos * weights.unsqueeze(-1), dim=1)
        valid_count = weights.sum(dim=1, keepdim=True)
        denom = torch.clamp(valid_count, min=1.0)
        fallback = self._default_particle_q.mean(dim=1) - self.scene.env_origins
        return torch.where(valid_count > 0.0, weighted_sum / denom, fallback)

    def _scoopable_particle_centroid(self, particle_pos: torch.Tensor) -> torch.Tensor:
        particle_pos = self._sanitize_particle_pos(particle_pos)
        scoopable = self._particles_in_workspace(particle_pos) & ~self._particles_in_bin(particle_pos)
        weights = scoopable.float()
        weighted_sum = torch.sum(particle_pos * weights.unsqueeze(-1), dim=1)
        valid_count = weights.sum(dim=1, keepdim=True)
        fallback = self._robust_particle_centroid(particle_pos)
        return torch.where(valid_count > 0.0, weighted_sum / torch.clamp(valid_count, min=1.0), fallback)

    def _count_particles_in_bin(self, particle_pos: torch.Tensor) -> torch.Tensor:
        return self._particles_in_bin(particle_pos).sum(dim=1, dtype=torch.float32)

    def _particles_in_bin(self, particle_pos: torch.Tensor) -> torch.Tensor:
        finite = torch.isfinite(particle_pos).all(dim=-1)
        above_lower = particle_pos > self._bin_lower
        below_upper = particle_pos < self._bin_upper
        return finite & torch.all(above_lower & below_upper, dim=-1)

    def _particles_in_workspace(self, particle_pos: torch.Tensor) -> torch.Tensor:
        finite = torch.isfinite(particle_pos).all(dim=-1)
        above_lower = particle_pos > self._workspace_lower
        below_upper = particle_pos < self._workspace_upper
        return finite & torch.all(above_lower & below_upper, dim=-1)

    def _particles_spilled(self, particle_pos: torch.Tensor) -> torch.Tensor:
        return ~self._particles_in_workspace(particle_pos)

    def _particle_bin_proximity(self, particle_pos: torch.Tensor) -> torch.Tensor:
        in_workspace = self._particles_in_workspace(particle_pos).float()
        xy_scale = torch.clamp(self._bin_half_extents[:2], min=1.0e-6)
        z_scale = max(float(self.cfg.bin_wall_height), 1.0e-6)
        xy_error = (particle_pos[..., :2] - self._bin_target[:2]) / xy_scale
        z_error = (particle_pos[..., 2] - self._bin_target[2]) / z_scale
        distance = torch.sqrt(torch.sum(torch.square(xy_error), dim=-1) + 0.25 * torch.square(z_error))
        score = torch.exp(-torch.clamp(distance, max=4.0))
        return (score * in_workspace).sum(dim=1) / torch.clamp(in_workspace.sum(dim=1), min=1.0)

    def _particle_progress_toward_bin(self, particle_pos: torch.Tensor) -> torch.Tensor:
        particle_x = particle_pos[..., 0]
        progress_start_x = self._current_pile_center[:, 0:1] - self._current_pile_half_extents[:, 0:1]
        progress_range = torch.clamp(self._progress_target_x - progress_start_x, min=1.0e-6)
        x_progress = (particle_x - progress_start_x) / progress_range
        x_progress = torch.clamp(x_progress, 0.0, 1.0)
        y_error = (particle_pos[..., 1] - self._bin_center[1]) / max(float(self.cfg.bin_inner_half_extents[1]), 1.0e-6)
        y_alignment = torch.exp(-torch.square(y_error))
        z_valid = (particle_pos[..., 2] > self._heightmap_z_min) & (particle_pos[..., 2] < self._workspace_upper[2])
        return (x_progress * y_alignment * z_valid.float()).mean(dim=1)

    def _particle_centroid_progress_toward_bin(self, particle_centroid: torch.Tensor) -> torch.Tensor:
        progress_start_x = self._current_pile_center[:, 0] - self._current_pile_half_extents[:, 0]
        progress_range = torch.clamp(self._progress_target_x - progress_start_x, min=1.0e-6)
        x_progress = torch.clamp((particle_centroid[:, 0] - progress_start_x) / progress_range, 0.0, 1.0)
        y_error = (particle_centroid[:, 1] - self._bin_center[1]) / max(
            float(self.cfg.bin_inner_half_extents[1]), 1.0e-6
        )
        y_alignment = torch.exp(-torch.square(y_error))
        return x_progress * y_alignment

    def _particle_mouth_entry(self, particle_pos: torch.Tensor) -> torch.Tensor:
        mouth_depth = max(0.08, 2.0 * self.cfg.voxel_size)
        in_mouth_x = torch.abs(particle_pos[..., 0] - self._bin_mouth_center[0]) < mouth_depth
        in_mouth_y = torch.abs(particle_pos[..., 1] - self._bin_mouth_center[1]) < self._bin_half_extents[1]
        near_entry_height = torch.abs(particle_pos[..., 2] - self._bin_mouth_center[2]) < 0.18
        return (in_mouth_x & in_mouth_y & near_entry_height).float().mean(dim=1)

    def _target_success_fraction(self) -> float:
        if not self.cfg.curriculum_enabled:
            return float(self.cfg.success_fraction)
        stage = min(self._curriculum_stage, len(self.cfg.curriculum_stage_success_fractions) - 1)
        return float(self.cfg.curriculum_stage_success_fractions[stage])

    def _update_curriculum(
        self, success_rate: float, nonfinite_rate: float, bin_fraction: float, reset_count: int
    ) -> None:
        if not self.cfg.curriculum_enabled:
            return
        alpha = float(self.cfg.curriculum_success_ema_alpha)
        self._curriculum_success_ema = (1.0 - alpha) * self._curriculum_success_ema + alpha * success_rate
        self._curriculum_bin_fraction_ema = (1.0 - alpha) * self._curriculum_bin_fraction_ema + alpha * bin_fraction
        self._curriculum_resets_in_stage += reset_count
        max_stage = len(self.cfg.curriculum_stage_success_fractions) - 1
        if (
            nonfinite_rate > self.cfg.curriculum_decrease_nonfinite_rate
            and self._curriculum_stage > 0
            and self._curriculum_resets_in_stage >= self.cfg.curriculum_min_resets_per_stage
        ):
            self._curriculum_stage -= 1
            self._curriculum_success_ema = 0.0
            self._curriculum_bin_fraction_ema = 0.0
            self._curriculum_resets_in_stage = 0
            return
        if nonfinite_rate > self.cfg.curriculum_max_nonfinite_rate:
            self._curriculum_success_ema = 0.5 * self._curriculum_success_ema
            self._curriculum_bin_fraction_ema = 0.5 * self._curriculum_bin_fraction_ema
            return
        if self._curriculum_stage >= max_stage:
            return
        threshold = float(self.cfg.curriculum_success_rate_thresholds[self._curriculum_stage])
        target_fraction = float(self.cfg.curriculum_stage_success_fractions[self._curriculum_stage])
        min_bin_fraction = self.cfg.curriculum_min_bin_fraction_ratio * target_fraction
        if (
            self._curriculum_resets_in_stage >= self.cfg.curriculum_min_resets_per_stage
            and self._curriculum_success_ema >= threshold
            and max(bin_fraction, self._curriculum_bin_fraction_ema) >= min_bin_fraction
        ):
            self._curriculum_stage += 1
            self._curriculum_success_ema = 0.0
            self._curriculum_bin_fraction_ema = 0.0
            self._curriculum_resets_in_stage = 0

    def _sample_particle_reset_positions(self, env_ids: torch.Tensor) -> torch.Tensor:
        if not self.cfg.curriculum_enabled:
            self._current_pile_center[env_ids] = self._default_pile_center
            self._current_pile_half_extents[env_ids] = self._default_pile_half_extents
            return self._default_particle_q[env_ids]

        stage = min(self._curriculum_stage, len(self.cfg.curriculum_stage_success_fractions) - 1)
        x_range = self.cfg.curriculum_pile_center_x_ranges[stage]
        y_range = self.cfg.curriculum_pile_center_y_ranges[stage]
        scale_range = self.cfg.curriculum_pile_scale_ranges[stage]
        num_resets = env_ids.numel()
        rand = torch.rand(num_resets, 3, device=self.device)
        center = self._default_pile_center.unsqueeze(0).repeat(num_resets, 1)
        center[:, 0] = float(x_range[0]) + rand[:, 0] * float(x_range[1] - x_range[0])
        center[:, 1] = float(y_range[0]) + rand[:, 1] * float(y_range[1] - y_range[0])
        center[:, 2] = 0.5 * (self.cfg.pile_lo[2] + self.cfg.pile_hi[2])
        scale = float(scale_range[0]) + rand[:, 2:3] * float(scale_range[1] - scale_range[0])
        scaled_offsets = self._default_particle_offsets[env_ids] * scale.unsqueeze(-1)
        local_particle_pos = center[:, None, :] + scaled_offsets
        table_margin = max(0.5 * float(self.cfg.voxel_size), 0.01)
        local_particle_pos[..., :2] = torch.clamp(
            local_particle_pos[..., :2],
            min=self._table_xy_lower + table_margin,
            max=self._table_xy_upper - table_margin,
        )
        local_particle_pos[..., 2] = torch.clamp(local_particle_pos[..., 2], min=self.cfg.table_top_z + 0.015)
        actual_pile_min = local_particle_pos.amin(dim=1)
        actual_pile_max = local_particle_pos.amax(dim=1)
        self._current_pile_center[env_ids] = 0.5 * (actual_pile_min + actual_pile_max)
        self._current_pile_half_extents[env_ids] = 0.5 * (actual_pile_max - actual_pile_min)
        return local_particle_pos + self.scene.env_origins[env_ids, None, :]

    def _reset_robot_to_curriculum_pose(self, env_ids: torch.Tensor, state_0, state_1, control) -> None:
        if not self.cfg.curriculum_enabled or not self.cfg.curriculum_robot_init_enabled:
            return

        stage = min(self._curriculum_stage, len(self.cfg.curriculum_stage_success_fractions) - 1)
        num_resets = env_ids.numel()
        rand = torch.rand(num_resets, 2, device=self.device)
        x_offsets = self.cfg.curriculum_robot_start_x_offset_ranges[stage]
        y_noise = self.cfg.curriculum_robot_start_y_noise_ranges[stage]
        target_paddle = self._current_pile_center[env_ids].clone()
        edge_offset = self._current_pile_half_extents[env_ids, 0]
        target_paddle[:, 0] -= edge_offset + float(x_offsets[0]) + rand[:, 0] * float(x_offsets[1] - x_offsets[0])
        target_paddle[:, 1] += float(y_noise[0]) + rand[:, 1] * float(y_noise[1] - y_noise[0])
        paddle_xy_margin = self._paddle_half_size[:2] + 0.02
        paddle_xy_lower = self._table_xy_lower + paddle_xy_margin
        paddle_xy_upper = self._table_xy_upper - paddle_xy_margin
        target_paddle[:, 1] = torch.clamp(target_paddle[:, 1], min=paddle_xy_lower[1], max=paddle_xy_upper[1])
        behind_pile_x = self._current_pile_center[env_ids, 0] - edge_offset - float(x_offsets[0])
        paddle_x_upper = torch.maximum(torch.minimum(paddle_xy_upper[0], behind_pile_x), paddle_xy_lower[0])
        target_paddle[:, 0] = torch.clamp(target_paddle[:, 0], min=paddle_xy_lower[0], max=paddle_x_upper)
        target_paddle[:, 2] = max(
            self.cfg.table_top_z + float(self.cfg.curriculum_robot_start_z_offsets[stage]),
            float(self.cfg.paddle_center_min_height),
        )
        target_paddle = self._clamp_paddle_target_position(target_paddle)

        model = NewtonManager.get_model()
        joint_q = wp.to_torch(state_0.joint_q)
        joint_qd = wp.to_torch(state_0.joint_qd)
        eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0, None)

        target_paddle_all, target_quat_all = self._paddle_pose_e()
        target_paddle_all = target_paddle_all.clone()
        target_quat_all = target_quat_all.clone()
        target_paddle_all[env_ids] = target_paddle
        target_quat_all[env_ids] = self._target_paddle_quat()[env_ids]
        current_joint_q = joint_q[self._joint_q_ids].clone()
        max_reset_delta = self.cfg.max_ik_reset_delta_q * self.cfg.curriculum_robot_init_iterations
        solved_joint_q = self._solve_newton_ik(
            current_joint_q,
            target_paddle_all,
            target_quat_all,
            self.cfg.ik_reset_iterations,
            max_reset_delta,
        )
        joint_q[self._joint_q_ids[env_ids]] = solved_joint_q[env_ids]
        joint_qd[self._joint_qd_ids[env_ids]] = 0.0
        eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0, None)

        achieved_paddle = self._paddle_pos_e()[env_ids]
        reset_error = torch.linalg.norm(achieved_paddle - target_paddle, dim=-1)
        reset_failed = (~torch.isfinite(reset_error)) | (reset_error > self.cfg.reset_ik_position_tolerance)
        self._last_reset_ik_position_error[env_ids] = torch.nan_to_num(
            reset_error,
            nan=self.cfg.reset_ik_position_tolerance,
            posinf=self.cfg.reset_ik_position_tolerance,
            neginf=self.cfg.reset_ik_position_tolerance,
        )
        self._last_reset_ik_failure[env_ids] = reset_failed
        if torch.any(reset_failed):
            failed_env_ids = env_ids[reset_failed]
            joint_q[self._joint_q_ids[failed_env_ids]] = current_joint_q[failed_env_ids]
            joint_qd[self._joint_qd_ids[failed_env_ids]] = 0.0
            eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0, None)

        wp.to_torch(state_1.joint_q)[self._joint_q_ids[env_ids]] = joint_q[self._joint_q_ids[env_ids]]
        wp.to_torch(state_1.joint_qd)[self._joint_qd_ids[env_ids]] = 0.0
        wp.to_torch(control.joint_target_pos)[self._joint_qd_ids[env_ids]] = joint_q[self._joint_q_ids[env_ids]]

    def _privileged_observations(self, particle_pos: torch.Tensor, paddle_velocity: torch.Tensor) -> torch.Tensor:
        stage_count = max(len(self.cfg.curriculum_stage_success_fractions) - 1, 1)
        stage = torch.full((self.num_envs, 1), self._curriculum_stage / stage_count, device=self.device)
        success_fraction = torch.full((self.num_envs, 1), self._target_success_fraction(), device=self.device)
        success_ema = torch.full((self.num_envs, 1), self._curriculum_success_ema, device=self.device)
        pile_center = self._normalize_positions(self._current_pile_center)
        pile_scale = self._current_pile_half_extents / torch.clamp(self._default_pile_half_extents, min=1.0e-6)
        particle_std = torch.std(particle_pos, dim=1, unbiased=False) / torch.tensor(
            (self._heightmap_x_range, self._heightmap_y_range, self.cfg.heightmap_z_range), device=self.device
        )
        paddle_speed = torch.linalg.norm(paddle_velocity, dim=-1, keepdim=True) / self.cfg.max_paddle_speed
        return torch.cat(
            (stage, success_fraction, success_ema, pile_center, pile_scale, particle_std, paddle_speed), dim=-1
        )

    def _configure_newton_viewer(self) -> None:
        for visualizer in self.sim.visualizers:
            viewer = getattr(visualizer, "_viewer", None)
            if viewer is None:
                continue
            if hasattr(viewer, "show_particles"):
                viewer.show_particles = True
            if hasattr(viewer, "show_contacts"):
                viewer.show_contacts = True

    def _apply_viewer_forces(self) -> None:
        state = NewtonManager.get_state_0()
        for visualizer in self.sim.visualizers:
            viewer = getattr(visualizer, "_viewer", None)
            if viewer is not None and hasattr(viewer, "apply_forces"):
                viewer.apply_forces(state)
