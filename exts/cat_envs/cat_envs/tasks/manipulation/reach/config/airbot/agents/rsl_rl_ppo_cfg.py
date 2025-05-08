# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

# from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from cat_envs.tasks.utils.cleanrl.rl_cfg import CleanRlPpoActorCriticCfg

@configclass
class AirbotPPORunnerCfg(CleanRlPpoActorCriticCfg):
    num_steps = 24

    learning_rate = 1e-3
    num_iterations = 1000
    save_interval = 50
    gamma=0.99
    gae_lambda = 0.95
    updates_epochs =5
    minibatch_size= 24576
    clip_coef = 0.2
    ent_coef = 0.001
    vf_coef = 2.0
    max_grad_norm = 1.0
    norm_adv = True
    clip_vloss = True
    anneal_lr = True

    # experiment_name = "solo12_flat"
    logger = "tensorboard"
    wandb_project = "airbot"

    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    experiment_name = "airbot"
    # run_name = ""
    # resume = False
    # empirical_normalization = False
    # policy = RslRlPpoActorCriticCfg(
    #     init_noise_std=1.0,
    #     actor_hidden_dims=[64, 64],
    #     critic_hidden_dims=[64, 64],
    #     activation="elu",
    # )
    # algorithm = RslRlPpoAlgorithmCfg(
    #     value_loss_coef=1.0,
    #     use_clipped_value_loss=True,
    #     clip_param=0.2,
    #     entropy_coef=0.001,
    #     num_learning_epochs=8,
    #     num_mini_batches=4,
    #     learning_rate=1.0e-3,
    #     schedule="adaptive",
    #     gamma=0.99,
    #     lam=0.95,
    #     desired_kl=0.01,
    #     max_grad_norm=1.0,
    # )
