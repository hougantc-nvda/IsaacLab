# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym


gym.register(
    id="Isaac-Waterhose-RBY1DF-IK-Rel-v0",
    entry_point="isaaclab_tasks.manager_based.manipulation.waterhose.waterhose_env:RBY1DFWaterhoseEnv",
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.manipulation.waterhose.waterhose_env_cfg:RBY1DFWaterhoseEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Waterhose-RBY1DF-IK-Rel-Play-v0",
    entry_point="isaaclab_tasks.manager_based.manipulation.waterhose.waterhose_env:RBY1DFWaterhoseEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.waterhose_env_cfg:RBY1DFWaterhoseEnvCfg_PLAY",
    },
    disable_env_checker=True,
)
