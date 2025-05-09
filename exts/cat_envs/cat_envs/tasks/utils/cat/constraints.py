# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_position(
    env: ManagerBasedRLEnv,
    limit: float,
    names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    joint_ids, _ = robot.find_joints(names, preserve_order=True)
    cstr = torch.abs(data.joint_pos[:, joint_ids]) - limit
    return cstr

def joint_position_lower_upper(
    env: ManagerBasedRLEnv,
    lower_limits: list[float],
    upper_limits: list[float],
    names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = robot.data

    joint_ids, _ = robot.find_joints(names, preserve_order=True)
    joint_positions = data.joint_pos[:, joint_ids]  # [num_envs, num_joints]

    # 转为张量，shape: [1, num_joints]
    lower_tensor = torch.tensor(lower_limits, device=joint_positions.device).unsqueeze(0)
    upper_tensor = torch.tensor(upper_limits, device=joint_positions.device).unsqueeze(0)

    # 计算每个关节是否超出上下限（负值表示没超出）
    below_lower = lower_tensor - joint_positions
    above_upper = joint_positions - upper_tensor

    # 如果 joint 在范围内，两者都为负，取 max 得负数（合法）；
    # 如果超出了某一侧，max 得到正数（违规程度）
    cstr = torch.maximum(below_lower, above_upper)  # [num_envs, num_joints]

    return cstr

def joint_position_when_moving_forward(
    env: ManagerBasedRLEnv,
    limit: float,
    names: list[str],
    velocity_deadzone: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    joint_ids, _ = robot.find_joints(names, preserve_order=True)
    cstr = (
        torch.abs(data.joint_pos[:, joint_ids] - data.default_joint_pos[:, joint_ids])
        - limit
    )
    cstr *= (
        (
            torch.abs(env.command_manager.get_command("base_velocity")[:, 1])
            < velocity_deadzone
        )
        .float()
        .unsqueeze(1)
    )
    return cstr


def joint_torque(
    env: ManagerBasedRLEnv,
    limit: float,
    names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    joint_ids, _ = robot.find_joints(names, preserve_order=True)
    cstr = torch.abs(data.applied_torque[:, joint_ids]) - limit
    return cstr


def joint_torque_list(
    env: ManagerBasedRLEnv,
    limits: list[float],  
    names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    joint_ids, _ = robot.find_joints(names, preserve_order=True)
    
    # Get the device of the tensor we're working with
    device = data.applied_torque.device
    
    # Create the limits tensor on the same device
    limits_tensor = torch.tensor(limits, dtype=torch.float32, device=device)
    
    # Make sure limits_tensor has the right shape for broadcasting
    if len(limits) == len(joint_ids):
        # Reshape for broadcasting across batches
        limits_tensor = limits_tensor.reshape(1, -1)
    
    cstr = torch.abs(data.applied_torque[:, joint_ids]) - limits_tensor
    return cstr


def joint_velocity(
    env: ManagerBasedRLEnv,
    limit: float,
    names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    joint_ids, _ = robot.find_joints(names, preserve_order=True)
    return torch.abs(data.joint_vel[:, joint_ids]) - limit

def joint_velocity_list(
    env: ManagerBasedRLEnv,
    limits: list[float], 
    names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    joint_ids, _ = robot.find_joints(names, preserve_order=True)
    
    # Get the device of the tensor we're working with
    device = data.joint_vel.device
    
    # Create the limits tensor on the same device
    limits_tensor = torch.tensor(limits, dtype=torch.float32, device=device)
    
    # Make sure limits_tensor has the right shape for broadcasting
    if len(limits) == len(joint_ids):
        # Reshape for broadcasting across batches
        limits_tensor = limits_tensor.reshape(1, -1)
    
    return torch.abs(data.joint_vel[:, joint_ids]) - limits_tensor

def joint_acceleration(
    env: ManagerBasedRLEnv,
    limit: float,
    names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    joint_ids, _ = robot.find_joints(names, preserve_order=True)
    return torch.abs(data.joint_acc[:, joint_ids]) - limit


def upsidedown(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    data = env.scene[asset_cfg.name].data
    return data.projected_gravity_b[:, 2] > limit


def contact(
    env: ManagerBasedRLEnv,
    names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    contact_sensor = env.scene[asset_cfg.name]
    undesired_contact_body_ids, _ = contact_sensor.find_bodies(
        names, preserve_order=True
    )
    net_contact_forces = contact_sensor.data.net_forces_w_history
    return torch.any(
        torch.max(
            torch.norm(net_contact_forces[:, :, undesired_contact_body_ids], dim=-1),
            dim=1,
        )[0]
        > 1.0,
        dim=1,
    )


def base_orientation(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    data = env.scene[asset_cfg.name].data
    return torch.norm(data.projected_gravity_b[:, :2], dim=1) - limit


def air_time(
    env: ManagerBasedRLEnv,
    limit: float,
    names: list[str],
    velocity_deadzone: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    contact_sensor = env.scene[asset_cfg.name]
    feet_ids, _ = contact_sensor.find_bodies(names, preserve_order=True)
    touchdown = contact_sensor.compute_first_contact(env.step_dt)[:, feet_ids]
    last_air_time = contact_sensor.data.last_air_time[:, feet_ids]
    # Like in CaT
    command_more_than_limit = (
        (
            torch.norm(env.command_manager.get_command("base_velocity")[:, :3], dim=1)
            > velocity_deadzone
        )
        .float()
        .unsqueeze(1)
    )
    # Like in Isaaclab
    # command_more_than_limit = (
    #     torch.norm(env.command_manager.get_command("base_velocity")[:, :2], dim=1) > 0.1
    # )
    cstr = (limit - last_air_time) * touchdown.float() * command_more_than_limit
    return cstr


def n_foot_contact(
    env: ManagerBasedRLEnv,
    names: list[str],
    number_of_desired_feet: int,
    min_command_value: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    contact_sensor = env.scene[asset_cfg.name]
    undesired_contact_body_ids, _ = contact_sensor.find_bodies(
        names, preserve_order=True
    )
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_cstr = torch.abs(
        (
            torch.max(
                torch.norm(
                    net_contact_forces[:, :, undesired_contact_body_ids], dim=-1
                ),
                dim=1,
            )[0]
            > 1.0
        ).sum(1)
        - number_of_desired_feet
    )
    command_more_than_limit = (
        torch.norm(env.command_manager.get_command("base_velocity")[:, :3], dim=1)
        > min_command_value
    ).float()
    return contact_cstr * command_more_than_limit


def joint_range(
    env: ManagerBasedRLEnv,
    limit: float,
    names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    joint_ids, _ = robot.find_joints(names, preserve_order=True)
    return (
        torch.abs(data.joint_pos[:, joint_ids] - data.default_joint_pos[:, joint_ids])
        - limit
    )


def action_rate(
    env: ManagerBasedRLEnv,
    limit: float,
    names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    joint_ids, _ = robot.find_joints(names, preserve_order=True)
    return (
        torch.abs(
            env.action_manager._action[:, joint_ids]
            - env.action_manager._prev_action[:, joint_ids]
        )
        / env.step_dt
        - limit
    )


def foot_contact_force(
    env: ManagerBasedRLEnv,
    limit: float,
    names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    contact_sensor = env.scene[asset_cfg.name]
    feet_ids, _ = contact_sensor.find_bodies(names, preserve_order=True)
    net_contact_forces = contact_sensor.data.net_forces_w_history
    return (
        torch.max(torch.norm(net_contact_forces[:, :, feet_ids], dim=-1), dim=1)[0]
        - limit
    )


def min_base_height(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return limit - robot.data.root_pos_w[:, 2]


def no_move(
    env: ManagerBasedRLEnv,
    names: list[str],
    velocity_deadzone: float,
    joint_vel_limit: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    joint_ids, _ = robot.find_joints(names, preserve_order=True)
    cstr_nomove = (torch.abs(data.joint_vel[:, joint_ids]) - joint_vel_limit) * (
        torch.norm(env.command_manager.get_command("base_velocity")[:, :3], dim=1)
        < velocity_deadzone
    ).float().unsqueeze(1)
    return cstr_nomove
