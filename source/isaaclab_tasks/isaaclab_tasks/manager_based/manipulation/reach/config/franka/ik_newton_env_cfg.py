# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.envs.mdp.actions.newton_ik_actions_cfg import NewtonInverseKinematicsActionCfg
from isaaclab_newton.ik.newton_ik_manager_cfg import NewtonIKManagerCfg

from isaaclab.utils.configclass import configclass

from . import ik_rel_env_cfg


@configclass
class FrankaReachEnvCfg(ik_rel_env_cfg.FrankaReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # This task is intentionally Newton-native: Newton physics provides the
        # replicated prototype model consumed by the Newton IK action.
        self.sim.physics = self.sim.physics.newton_mjwarp
        # The reach objective does not interact with the table. Omitting it keeps
        # the Newton IK prototype model focused on the robot articulation.
        self.scene.table = None

        self.actions.arm_action = NewtonInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=NewtonIKManagerCfg(command_type="pose", use_relative_mode=True),
            scale=0.5,
            body_offset=NewtonInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )


@configclass
class FrankaReachEnvCfg_PLAY(FrankaReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
