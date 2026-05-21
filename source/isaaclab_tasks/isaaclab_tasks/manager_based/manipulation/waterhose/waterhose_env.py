# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.physics import PhysicsEvent

from . import waterhose_core as core
from .waterhose_env_cfg import RBY1DFWaterhoseEnvCfg


class RBY1DFWaterhoseEnv(ManagerBasedRLEnv):
    """Manager-based environment for the Newton proxy-coupled waterhose task."""

    cfg: RBY1DFWaterhoseEnvCfg

    def __init__(self, cfg: RBY1DFWaterhoseEnvCfg, render_mode: str | None = None, **kwargs):
        self.waterhose_scene_builder = None
        self.waterhose_builder = None
        self._waterhose_kit_scene_callback = None
        self._waterhose_cable_xform_callback = None
        self._build_waterhose_model(cfg)
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        self._finish_waterhose_runtime_setup()
        self.obs_buf = self.observation_manager.compute(update_history=True)

    def _build_waterhose_model(self, cfg: RBY1DFWaterhoseEnvCfg) -> None:
        self.waterhose_scene_builder, self.waterhose_builder, solver_cfg = core.build_waterhose_scene(
            **cfg.waterhose_scene_kwargs()
        )
        core.prefer_active_python_site_packages()
        cfg.sim.physics.solver_cfg = solver_cfg
        # NewtonCfg derives its manager class in __post_init__; keep it aligned
        # with the runtime-built coupled solver cfg.
        cfg.sim.physics.class_type = solver_cfg.class_type
        cfg.sync_waterhose_sim_cfg()
        core.NewtonManager.set_builder(self.waterhose_builder)
        self._register_kit_scene_callback()
        self._register_cable_xform_callback()

    def _register_kit_scene_callback(self) -> None:
        def setup_kit_scene(_payload) -> None:
            scene_builder = self.waterhose_scene_builder
            if scene_builder is None:
                return
            core.prepare_kit_scene_for_newton_sync(self.sim, scene_builder, self.waterhose_builder)
            handle = getattr(self, "_waterhose_kit_scene_callback", None)
            if handle is not None:
                handle.deregister()
                self._waterhose_kit_scene_callback = None

        self._waterhose_kit_scene_callback = core.NewtonManager.register_callback(
            setup_kit_scene,
            PhysicsEvent.MODEL_INIT,
            order=-1000,
            name="waterhose_kit_scene",
            wrap_weak_ref=False,
        )

    def reset(
        self, seed: int | None = None, env_ids: Sequence[int] | None = None, options: dict[str, Any] | None = None
    ):
        """Reset the environment and restore authored waterhose cable transforms."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)

        self.recorder_manager.record_pre_reset(env_ids)

        if seed is not None:
            self.seed(seed)

        self._reset_idx(env_ids)

        self.scene.write_data_to_sim()
        self.sim.forward()
        self._apply_waterhose_cable_asset_xform()

        if self.has_rtx_sensors and self.cfg.num_rerenders_on_reset > 0:
            for _ in range(self.cfg.num_rerenders_on_reset):
                self.sim.render()

        self.recorder_manager.record_post_reset(env_ids)

        self.obs_buf = self.observation_manager.compute(update_history=True)

        if self.cfg.wait_for_textures and self.has_rtx_sensors:
            if hasattr(self.sim.physics_manager, "assets_loading"):
                while self.sim.physics_manager.assets_loading():
                    self.sim.render()

        core.configure_kit_camera_view(self.sim)
        return self.obs_buf, self.extras

    def _register_cable_xform_callback(self) -> None:
        def apply_cable_xform(_payload) -> None:
            self.waterhose_scene_builder.apply_runtime_cable_asset_xform(sync_solver_prev=False)
            handle = getattr(self, "_waterhose_cable_xform_callback", None)
            if handle is not None:
                handle.deregister()
                self._waterhose_cable_xform_callback = None

        self._waterhose_cable_xform_callback = core.NewtonManager.register_callback(
            apply_cable_xform,
            PhysicsEvent.PHYSICS_READY,
            order=-1000,
            name="waterhose_cable_asset_xform",
            wrap_weak_ref=False,
        )

    def _finish_waterhose_runtime_setup(self) -> None:
        scene_builder = self.waterhose_scene_builder
        if scene_builder is None:
            raise RuntimeError("Waterhose scene builder was not initialized.")
        core.setup_kit_scene(self.sim, scene_builder)
        scene_builder.configure_runtime_vbd_solver()
        scene_builder.apply_runtime_cable_asset_xform()
        if self.sim.is_rendering:
            core.sync_kit_cable_curves_from_newton(scene_builder)
            core.NewtonManager.sync_transforms_to_usd()
        core.configure_newton_viewer(self.sim)
        core.configure_kit_camera_view(self.sim)
        self.waterhose_right_ee_body_ids = self._resolve_body_ids(core.RIGHT_EE)
        self.waterhose_tip_body_ids = self._resolve_matching_ids(
            core.NewtonManager.get_model().body_label[scene_builder.tip_body_id]
        )
        self.waterhose_plug_body_ids = self._resolve_matching_ids(
            core.NewtonManager.get_model().body_label[scene_builder.plug_body_id]
        )
        socket_pos = np.array([float(scene_builder.socket_pos[i]) for i in range(3)], dtype=np.float64)
        socket_rot = np.array([float(scene_builder.socket_rot[i]) for i in range(4)], dtype=np.float64)
        self.waterhose_socket_pose = np.concatenate((socket_pos, socket_rot), axis=0)

    def _resolve_body_ids(self, short_name: str) -> list[int]:
        labels = core.NewtonManager.get_model().body_label
        suffix = "/" + short_name
        matches = [idx for idx, label in enumerate(labels) if label == short_name or label.endswith(suffix)]
        if len(matches) < self.num_envs:
            raise RuntimeError(f"Expected at least {self.num_envs} bodies named {short_name!r}, found {matches}.")
        return matches[: self.num_envs]

    def _resolve_matching_ids(self, label: str) -> list[int]:
        labels = core.NewtonManager.get_model().body_label
        matches = [idx for idx, candidate in enumerate(labels) if candidate == label]
        if len(matches) < self.num_envs:
            # Some replicated builders uniquify labels; fall back to the primary id for single-env use.
            if self.num_envs == 1:
                return [labels.index(label)]
            raise RuntimeError(f"Expected {self.num_envs} bodies matching {label!r}, found {matches}.")
        return matches[: self.num_envs]

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        # The first implementation targets demonstration collection, where episodes
        # normally end after success. Full partial Newton state reset can be added
        # once the coupled solver exposes per-world reset hooks for VBD bodies.
        self._apply_waterhose_cable_asset_xform()

    def _apply_waterhose_cable_asset_xform(self) -> None:
        self.waterhose_scene_builder.apply_runtime_cable_asset_xform(sync_solver_prev=True)
        if self.sim.is_rendering:
            core.sync_kit_cable_curves_from_newton(self.waterhose_scene_builder)
            core.NewtonManager.sync_transforms_to_usd()

    def close(self) -> None:
        scene_builder = getattr(self, "waterhose_scene_builder", None)
        if scene_builder is not None:
            core.remove_kit_cable_curve_pre_render_sync(scene_builder)
        super().close()

    def get_task_space_action_term(self):
        return self.action_manager.get_term("task_space")

    def scripted_action(self) -> torch.Tensor:
        """Return a repeatable scripted task-space action for demonstration seeding."""
        if not hasattr(self, "_scripted_controller"):
            self._scripted_controller = core.WaterhoseIKController(self.waterhose_scene_builder)
            self._scripted_prev_target = None
            self._scripted_prev_quat = None
        controller = self._scripted_controller
        controller.state = core.NewtonManager.get_state_0()
        sim_time = float(self.common_step_counter) * self.step_dt
        while controller._should_advance_phase(sim_time):
            controller._enter_phase(min(controller.phase_index + 1, len(controller._PHASES) - 1), sim_time)
        target_pos, target_quat, gripper_value, phase = controller._target_for_phase(sim_time)
        target_pos, target_quat = controller._filter_ik_target(target_pos, target_quat)
        target_pos_t = torch.as_tensor([float(target_pos[i]) for i in range(3)], device=self.device)
        target_quat_t = torch.as_tensor([float(target_quat[i]) for i in range(4)], device=self.device)
        action = self.get_task_space_action_term().target_pose_to_action(
            target_pos_t.to(torch.float32),
            target_quat_t.to(torch.float32),
            env_id=0,
        )
        action[-1] = 1.0 if gripper_value > 0.5 else -1.0
        actions = action.unsqueeze(0).repeat(self.num_envs, 1)
        self.waterhose_last_scripted_phase = phase
        return actions
