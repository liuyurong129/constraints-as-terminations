# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from cat_envs.tasks.utils.cat.cat_env import CaTEnv
##
# Register Gym environments for Airbot
##

##
# Joint Position Control
##

gym.register(
    id="Airbot-v0",
    entry_point=CaTEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:AirbotEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "clean_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AirbotPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Airbot-Play-v0",
    entry_point=CaTEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:AirbotEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "clean_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AirbotPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="Airbot-IK-Abs-v0",
    entry_point=CaTEnv,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:AirbotEnvCfg",
        "clean_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AirbotPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Airbot-IK-Abs-Play-v0",
    entry_point=CaTEnv,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:AirbotEnvCfg_PLAY",
        "clean_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AirbotPPORunnerCfg",
    },
    disable_env_checker=True,
)
##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Airbot-IK-Rel-v0",
    entry_point=CaTEnv,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:AirbotEnvCfg",
        "clean_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AirbotPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Airbot-IK-Rel-Play-v0",
    entry_point=CaTEnv,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:AirbotEnvCfg_PLAY",
        "clean_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AirbotPPORunnerCfg",
    },
    disable_env_checker=True,
)
##
# Operational Space Control
##

gym.register(
    id="Airbot-OSC-v0",
    entry_point=CaTEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.osc_env_cfg:AirbotEnvCfg",
        "clean_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AirbotPPORunnerCfg",
    },
)

gym.register(
    id="Airbot-OSC-Play-v0",
    entry_point=CaTEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.osc_env_cfg:AirbotEnvCfg_PLAY",
        "clean_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AirbotPPORunnerCfg",
    },
)
