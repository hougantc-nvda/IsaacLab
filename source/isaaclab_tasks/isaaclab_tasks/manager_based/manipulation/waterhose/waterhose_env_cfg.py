# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

# Newton's USD importer is sensitive to import order, so initialize it before
# the broader Isaac Lab stack is imported below.  Kit launches are the
# exception: SimulationApp must start before Newton/PXR-facing imports.
from . import waterhose_core as core  # isort: skip
from .launch import should_defer_newton_import  # isort: skip

if not should_defer_newton_import():
    core.import_newton_dependencies()

from isaaclab_newton.physics import NewtonCfg

from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

from . import mdp
from .actions import NewtonTaskSpaceIKAction, NewtonTaskSpaceIKActionCfg


@configclass
class ActionsCfg:
    """Action terms for task-space waterhose control."""

    task_space = NewtonTaskSpaceIKActionCfg(class_type=NewtonTaskSpaceIKAction)


@configclass
class ObservationsCfg:
    """Observation terms for waterhose manipulation."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        eef_pos = ObsTerm(func=mdp.eef_pos)
        eef_quat = ObsTerm(func=mdp.eef_quat)
        plug_pos = ObsTerm(func=mdp.plug_pos)
        plug_quat = ObsTerm(func=mdp.plug_quat)
        tip_pos = ObsTerm(func=mdp.tip_pos)
        tip_quat = ObsTerm(func=mdp.tip_quat)
        socket_pose = ObsTerm(func=mdp.socket_pose)
        alignment = ObsTerm(func=mdp.alignment)
        proxy_wrench_norm = ObsTerm(func=mdp.proxy_wrench_norm)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        approach = ObsTerm(func=mdp.approach_done)
        grasp = ObsTerm(func=mdp.grasp_done)
        align = ObsTerm(func=mdp.align_done)
        insert = ObsTerm(func=mdp.insert_done)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class RewardsCfg:
    """Reward terms for the waterhose task."""

    reach_hose = RewTerm(func=mdp.reach_hose, weight=1.0)
    align_tip = RewTerm(func=mdp.align_tip, weight=2.0)
    insert_tip = RewTerm(func=mdp.insert_tip, weight=30.0)
    success_bonus = RewTerm(func=mdp.success_bonus, weight=20.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


@configclass
class TerminationsCfg:
    """Termination terms for the waterhose task."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.success)


@configclass
class EventCfg:
    """No reset events are needed; Newton owns the waterhose state."""

    pass


@configclass
class RBY1DFWaterhoseEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based Newton proxy-coupled RBY1 waterhose task."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None
    commands = None

    # The concrete coupled solver cfg is built from the waterhose assets in RBY1DFWaterhoseEnv.__init__.
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 100.0,
        render_interval=1,
        physics=NewtonCfg(num_substeps=10, use_cuda_graph=True),
    )
    viewer = ViewerCfg(eye=(-1.2, -2.8, 1.6), lookat=(0.55, -0.42, 0.55))

    episode_length_s = 30.0
    decimation = 1

    asset_root: str = f"{ISAACLAB_ASSETS_DATA_DIR}/Props/Waterhose"
    robot_urdf: str | None = None
    scene_usd: str | None = None
    cable_usds: str | None = None
    cable_usd: str | None = None
    cable_prims: str | None = None
    cable_prim: str | None = None
    sim_substeps: int = 10
    rigid_substeps: int = 1
    proxy_iterations: int = 1
    proxy_mass_scale: float = 1.0
    vbd_iterations: int = 10
    rigid_contact_max: int = 100000
    mujoco_iterations: int = 20
    mujoco_ls_iterations: int = 10
    mujoco_ls_parallel: bool = True
    mujoco_impratio: float = 1000.0
    mujoco_use_mujoco_contacts: bool = True
    disable_cuda_graph: bool = False
    hose_radius: float = 0.003
    gripper_drive_scale: float = 2.0
    robot_shape_margin: float = 0.0
    robot_shape_gap: float = 0.005
    robot_shape_ke: float = 5.0e4
    robot_shape_kd: float = 5.0e2
    robot_shape_mu: float = 2.0
    robot_joint_target_ke: float = 45000.0
    robot_joint_target_kd: float = 4500.0
    robot_joint_effort_limit: float = 1000.0
    robot_joint_armature: float = 0.2
    gripper_joint_target_ke: float = 10000.0
    gripper_joint_target_kd: float = 1000.0
    gripper_joint_effort_limit: float = 100000.0
    gripper_joint_armature: float = 0.5
    grasp_friction: float = 3.0e6
    grasp_margin: float = 0.001
    grasp_contact_ke: float = 2.0e5
    vbd_collide_substeps: int = 5
    vbd_default_contact_ke: float = 1.0e5
    vbd_default_contact_kd: float = 1.0e-1
    vbd_default_contact_margin: float = 0.001
    vbd_solver_friction_epsilon: float = 0.1
    vbd_rigid_contact_buffer_size: int = 2048
    vbd_proxy_margin: float = 0.001
    vbd_cable_density: float = 10000.0
    vbd_cable_mu: float = 1.0
    vbd_cable_margin: float = 0.0
    vbd_cable_gap: float = 0.001
    vbd_static_margin: float = 1.0e-4
    vbd_static_gap: float = 0.001
    vbd_near_tip_mu: float = 1.0e1
    vbd_far_tip_mu: float = 1.0e5
    vbd_ground_mu: float = 1.0e5
    vbd_rigid_avbd_beta: float = 1.0e5
    vbd_rigid_contact_k_start: float = 1.0e2
    vbd_rigid_joint_linear_k_start: float = 1.0e4
    vbd_rigid_joint_angular_k_start: float = 1.0e1
    cable_stretch_stiffness: float = 1.0e12
    cable_stretch_damping: float = 1.0e-3
    cable_num_segments: int = 100
    cable_bend_rigidity: float = 3.0e0
    cable_bend_damping: float = 1.0e0
    success_lateral_threshold: float = 0.0008
    success_axis_cosine: float = -0.995
    success_insert_depth: float = 0.025
    insert_start_depth: float = 0.005

    def __post_init__(self):
        self.scene.num_envs = max(1, int(self.scene.num_envs))
        self.sync_waterhose_sim_cfg()

    def sync_waterhose_sim_cfg(self) -> None:
        """Synchronize derived Newton simulation settings from task cfg fields."""
        self.sim.render_interval = self.decimation
        self.sim.physics.num_substeps = int(self.sim_substeps)
        self.sim.physics.use_cuda_graph = not bool(self.disable_cuda_graph)

    def waterhose_scene_kwargs(self) -> dict[str, object]:
        """Return arguments used to build the Newton waterhose scene."""
        return {
            "fps": 1.0 / float(self.sim.dt),
            "num_envs": int(self.scene.num_envs),
            "env_spacing": float(self.scene.env_spacing),
            "asset_root": self.asset_root,
            "robot_urdf": self.robot_urdf,
            "scene_usd": self.scene_usd,
            "cable_usds": self.cable_usds,
            "cable_usd": self.cable_usd,
            "cable_prims": self.cable_prims,
            "cable_prim": self.cable_prim,
            "hose_radius": float(self.hose_radius),
            "gripper_drive_scale": float(self.gripper_drive_scale),
            "grasp_friction": float(self.grasp_friction),
            "grasp_margin": float(self.grasp_margin),
            "grasp_contact_ke": float(self.grasp_contact_ke),
            "sim_substeps": int(self.sim_substeps),
            "rigid_substeps": int(self.rigid_substeps),
            "proxy_iterations": int(self.proxy_iterations),
            "proxy_mass_scale": float(self.proxy_mass_scale),
            "vbd_iterations": int(self.vbd_iterations),
            "rigid_contact_max": int(self.rigid_contact_max),
            "mujoco_iterations": int(self.mujoco_iterations),
            "mujoco_ls_iterations": int(self.mujoco_ls_iterations),
            "mujoco_ls_parallel": bool(self.mujoco_ls_parallel),
            "mujoco_impratio": float(self.mujoco_impratio),
            "mujoco_use_mujoco_contacts": bool(self.mujoco_use_mujoco_contacts),
            "robot_shape_margin": float(self.robot_shape_margin),
            "robot_shape_gap": float(self.robot_shape_gap),
            "robot_shape_ke": float(self.robot_shape_ke),
            "robot_shape_kd": float(self.robot_shape_kd),
            "robot_shape_mu": float(self.robot_shape_mu),
            "robot_joint_target_ke": float(self.robot_joint_target_ke),
            "robot_joint_target_kd": float(self.robot_joint_target_kd),
            "robot_joint_effort_limit": float(self.robot_joint_effort_limit),
            "robot_joint_armature": float(self.robot_joint_armature),
            "gripper_joint_target_ke": float(self.gripper_joint_target_ke),
            "gripper_joint_target_kd": float(self.gripper_joint_target_kd),
            "gripper_joint_effort_limit": float(self.gripper_joint_effort_limit),
            "gripper_joint_armature": float(self.gripper_joint_armature),
            "vbd_collide_substeps": int(self.vbd_collide_substeps),
            "vbd_default_contact_ke": float(self.vbd_default_contact_ke),
            "vbd_default_contact_kd": float(self.vbd_default_contact_kd),
            "vbd_default_contact_margin": float(self.vbd_default_contact_margin),
            "vbd_solver_friction_epsilon": float(self.vbd_solver_friction_epsilon),
            "vbd_rigid_contact_buffer_size": int(self.vbd_rigid_contact_buffer_size),
            "vbd_proxy_margin": float(self.vbd_proxy_margin),
            "vbd_cable_density": float(self.vbd_cable_density),
            "vbd_cable_mu": float(self.vbd_cable_mu),
            "vbd_cable_margin": float(self.vbd_cable_margin),
            "vbd_cable_gap": float(self.vbd_cable_gap),
            "vbd_static_margin": float(self.vbd_static_margin),
            "vbd_static_gap": float(self.vbd_static_gap),
            "vbd_near_tip_mu": float(self.vbd_near_tip_mu),
            "vbd_far_tip_mu": float(self.vbd_far_tip_mu),
            "vbd_ground_mu": float(self.vbd_ground_mu),
            "vbd_rigid_avbd_beta": float(self.vbd_rigid_avbd_beta),
            "vbd_rigid_contact_k_start": float(self.vbd_rigid_contact_k_start),
            "vbd_rigid_joint_linear_k_start": float(self.vbd_rigid_joint_linear_k_start),
            "vbd_rigid_joint_angular_k_start": float(self.vbd_rigid_joint_angular_k_start),
            "cable_stretch_stiffness": float(self.cable_stretch_stiffness),
            "cable_stretch_damping": float(self.cable_stretch_damping),
            "cable_num_segments": int(self.cable_num_segments),
            "cable_bend_rigidity": float(self.cable_bend_rigidity),
            "cable_bend_damping": float(self.cable_bend_damping),
            "disable_cuda_graph": bool(self.disable_cuda_graph),
            "device": str(self.sim.device),
        }
