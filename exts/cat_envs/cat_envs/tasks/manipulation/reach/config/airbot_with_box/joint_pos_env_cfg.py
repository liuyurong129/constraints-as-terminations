# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets import AIRBOT_BOX_CFG  # isort: skip
import cat_envs.tasks.utils.cat.constraints as constraints
from cat_envs.tasks.utils.cat.manager_constraint_cfg import (
    ConstraintTermCfg as ConstraintTerm,
)


@configclass
class ConstraintsCfg:
    # Safety Soft constraints

    joint_position=ConstraintTerm(
        func=constraints.joint_position_lower_upper,
        max_p=1.0,
        params={
            "lower_limits": [-3.14, -2.96, -0.087, -2.96, -1.74, -3.14],
            "upper_limits": [ 2.09,  0.17,  3.14,  2.96,  1.74,  3.14],
            "names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        }
    )

    joint_velocity = ConstraintTerm(
        func=constraints.joint_velocity_list,
        max_p=0.25,
        params={
            "limits": [3.14, 3.14, 3.14, 6.28, 6.28, 6.28],
            "names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        },
    )

    joint_torque = ConstraintTerm(
        func=constraints.joint_torque_list,
        max_p=0.1,
        params={
            "limits": [18.0, 18.0, 18.0, 3.0, 3.0, 3.0], 
            "names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        },
    )

    joint_acceleration = ConstraintTerm(
        func=constraints.joint_acceleration,
        max_p=0.25,
        params={
            "limit": 10.0,  # Estimated value based on common robot arm specifications
            "names": ["joint.*"],
        },
    )

    action_rate = ConstraintTerm(
        func=constraints.action_rate,
        max_p=0.25,
        params={
            "limit": 80.0,  # Limits how fast actions can change to prevent jerky motion
            "names": ["joint.*"],
        },
    )

##
# Environment configuration
##


@configclass
class AirbotBoxEnvCfg(ReachEnvCfg):
    constraints: ConstraintsCfg = ConstraintsCfg()
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to franka
        self.scene.robot = AIRBOT_BOX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["ee_link"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["ee_link"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["ee_link"]
        self.rewards.end_effector_position_tracking.weight = -0.2
        self.rewards.end_effector_position_tracking_fine_grained.weight = 0.5
        self.rewards.end_effector_orientation_tracking.weight = -0.1
        # self.rewards.action_rate.weight = -0.001

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["joint.*"], scale=0.5, use_default_offset=True
        )
        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = "ee_link"
        # self.commands.ee_pose.ranges.roll = (0,0)
        # self.commands.ee_pose.ranges.pitch = (math.pi/2,math.pi/2)
        # self.commands.ee_pose.ranges.yaw = (-math.pi/2, math.pi/2)
        self.commands.ee_pose.ranges.roll = (math.pi/2, math.pi/2)
        self.commands.ee_pose.ranges.pitch = (-math.pi/2, math.pi/2)
        self.commands.ee_pose.ranges.yaw = (math.pi/2, math.pi/2)
        # self.commands.ee_pose.ranges.pos_x = (0.75, 0.8)
        self.commands.ee_pose.ranges.pos_z = (0.2, 0.5)


@configclass
class AirbotBoxEnvCfg_PLAY(AirbotBoxEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
