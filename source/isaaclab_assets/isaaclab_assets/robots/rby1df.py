# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Rainbow Robotics RBY1DF mobile manipulator.

The asset is imported from the vendored URDF in ``isaaclab_assets/data``.  The
first training tasks use the robot as a fixed-base manipulator and control only
one arm; the remaining joints are still actuated so they hold their default pose.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

RBY1DF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/RBY1DF/urdf/robot_edited.urdf",
        usd_dir="/tmp/IsaacLab/RBY1DF",
        usd_file_name="rby1df.usd",
        fix_base=False,
        merge_fixed_joints=True,
        make_instanceable=True,
        self_collision=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "torso_joint_1": 0.048646115,
            "torso_joint_2": -0.11358134,
            "torso_joint_3": 0.28509942,
            "torso_joint_4": 0.30236751,
            "torso_joint_5": -0.043634601,
            "torso_joint_6": 0.009673167,
            "right_arm_joint_1": -0.85306484,
            "right_arm_joint_2": -1.0891527,
            "right_arm_joint_3": 0.66765565,
            "right_arm_joint_4": -2.0121396,
            "right_arm_joint_5": -1.0203781,
            "right_arm_joint_6": 1.5501461,
            "right_arm_joint_7": 0.56562239,
            "left_arm_joint_1": -0.70531148,
            "left_arm_joint_2": 1.0506693,
            "left_arm_joint_3": -0.44851208,
            "left_arm_joint_4": -1.9159117,
            "left_arm_joint_5": 1.0035634,
            "left_arm_joint_6": 1.5637023,
            "left_arm_joint_7": -0.84481186,
            "head_joint_.*": 0.0,
            "left_gripper_finger_joint_1": 0.08,
            "left_gripper_left_finger_joint": -0.04,
            "left_gripper_right_finger_joint": 0.04,
            "right_gripper_finger_joint_1": 0.08,
            "right_gripper_left_finger_joint": -0.04,
            "right_gripper_right_finger_joint": 0.04,
        },
    ),
    actuators={
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["torso_joint_[1-6]"],
            effort_limit_sim={
                "torso_joint_[1-3]": 270.0,
                "torso_joint_[4-6]": 120.0,
            },
            velocity_limit_sim={
                "torso_joint_[1-3]": 2.09439510,
                "torso_joint_[4-6]": 3.14159265,
            },
            stiffness=300.0,
            damping=30.0,
            armature=1e-2,
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["right_arm_joint_[1-7]"],
            effort_limit_sim={
                "right_arm_joint_[1-3]": 70.0,
                "right_arm_joint_4": 40.0,
                "right_arm_joint_[5-6]": 10.0,
                "right_arm_joint_7": 8.0,
            },
            velocity_limit_sim={
                "right_arm_joint_[1-4]": 3.14159265,
                "right_arm_joint_[5-6]": 6.283185308,
                "right_arm_joint_7": 2.094395102,
            },
            stiffness=120.0,
            damping=8.0,
            armature=1e-3,
        ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["left_arm_joint_[1-7]"],
            effort_limit_sim={
                "left_arm_joint_[1-3]": 70.0,
                "left_arm_joint_4": 40.0,
                "left_arm_joint_[5-6]": 10.0,
                "left_arm_joint_7": 8.0,
            },
            velocity_limit_sim={
                "left_arm_joint_[1-4]": 3.14159265,
                "left_arm_joint_[5-6]": 6.283185308,
                "left_arm_joint_7": 2.094395102,
            },
            stiffness=120.0,
            damping=8.0,
            armature=1e-3,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_joint_[1-2]"],
            effort_limit_sim=1000.0,
            velocity_limit_sim=3.14,
            stiffness=50.0,
            damping=5.0,
        ),
        "grippers": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_gripper_.*finger_joint.*",
                "right_gripper_.*finger_joint.*",
            ],
            effort_limit_sim=200.0,
            velocity_limit_sim=1.0,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration of the RBY1DF robot imported from URDF."""


RBY1DF_FIXED_BASE_CFG = RBY1DF_CFG.copy()
RBY1DF_FIXED_BASE_CFG.spawn.fix_base = True
RBY1DF_FIXED_BASE_CFG.spawn.usd_file_name = "rby1df_fixed_base.usd"
"""Configuration of the RBY1DF robot imported from URDF with a fixed base."""


RBY1DF_HIGH_PD_CFG = RBY1DF_CFG.copy()
RBY1DF_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
for _actuator_name in ("torso", "right_arm", "left_arm", "head"):
    RBY1DF_HIGH_PD_CFG.actuators[_actuator_name].stiffness = 45000.0
    RBY1DF_HIGH_PD_CFG.actuators[_actuator_name].damping = 4500.0
    RBY1DF_HIGH_PD_CFG.actuators[_actuator_name].effort_limit_sim = 1000.0
    RBY1DF_HIGH_PD_CFG.actuators[_actuator_name].armature = 0.2
RBY1DF_HIGH_PD_CFG.actuators["grippers"].stiffness = 10000.0
RBY1DF_HIGH_PD_CFG.actuators["grippers"].damping = 1000.0
RBY1DF_HIGH_PD_CFG.actuators["grippers"].effort_limit_sim = 100000.0
RBY1DF_HIGH_PD_CFG.actuators["grippers"].armature = 0.5
"""Configuration of the RBY1DF robot with stiffer PD gains for task-space IK."""


RBY1DF_FIXED_BASE_HIGH_PD_CFG = RBY1DF_HIGH_PD_CFG.copy()
RBY1DF_FIXED_BASE_HIGH_PD_CFG.spawn.fix_base = True
RBY1DF_FIXED_BASE_HIGH_PD_CFG.spawn.usd_file_name = "rby1df_fixed_base_high_pd.usd"
"""Configuration of the fixed-base RBY1DF robot with stiffer PD gains for task-space IK."""
