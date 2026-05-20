# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.manipulation.waterhose.waterhose_env_cfg import RBY1DFWaterhoseEnvCfg


@configclass
class RBY1DFWaterhoseEnvCfg_PLAY(RBY1DFWaterhoseEnvCfg):
    """Small play variant for Newton viewer and teleoperation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.sync_waterhose_sim_cfg()
        self.actions.task_space.accumulate_targets = True
        self.actions.task_space.command_frame = "world"
        self.actions.task_space.rotation_frame = "eef"
        self.observations.policy.enable_corruption = False
