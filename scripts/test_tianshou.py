import argparse
import datetime
import logging
import os
import pprint

import gymnasium as gym
import hydra
import minigrid
from minigrid.wrappers import (
    FullyObsWrapper,
    ImgObsWrapper,
    SymbolicObsWrapper,
)
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.tensorboard import SummaryWriter
import tqdm

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.policy import SACPolicy, DiscreteSACPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.env.venvs import DummyVectorEnv

import stable_baselines3.common.callbacks as cb
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecMonitor,
    DummyVecEnv,
    VecVideoRecorder,
)
from belief_srs.envs.gridworld import POMDPGridWorld
from belief_srs.utils.wrappers import RGBImgObsWrapper, ProcessImageWrapper
from belief_srs.utils.layers import NatureCNN
from belief_srs.utils.tianshou_wandb import WandbLogger
import wandb

logger = logging.getLogger(__name__)


def make_env(cfg):
    render_mode = "human" if cfg.render else "rgb_array"
    device = torch.device(0) if torch.cuda.device_count() else torch.device("cpu")

    def _make_env(seed, rank=0, test=False):
        env_cfg = cfg.env[cfg.env.name]
        if cfg.env.name == "gridworld":
            if env_cfg.full_obs:
                # agent view is irrelevant
                env_cfg.highlight = False

            logger.info(env_cfg)

            env = POMDPGridWorld(
                size=env_cfg.size,
                agent_start_pos=env_cfg.agent_start_pos,
                agent_start_dir=env_cfg.agent_start_dir,
                agent_view_size=env_cfg.agent_view_size,
                max_steps=env_cfg.max_steps,
                render_mode=render_mode,
                highlight=env_cfg.highlight,
            )
            env.reset(seed=seed + rank)

            if env_cfg.obs.type == "symbolic":
                env = SymbolicObsWrapper(env)

            elif env_cfg.obs.type == "sensory":
                env = RGBImgObsWrapper(
                    env,
                    highlight=env_cfg.highlight,
                    partial_view=not (env_cfg.full_obs),
                )

            else:
                raise NotImplementedError()

            # don't need language obs
            env = ImgObsWrapper(env)
            env = ProcessImageWrapper(env)

        else:
            raise NotImplementedError()

        return env

    if cfg.env.name == "gridworld":
        env = DummyVectorEnv(
            [lambda: _make_env(cfg.seed, rank=i) for i in range(cfg.rl.num_envs)]
        )
    else:
        env = DummyVectorEnv(
            [lambda: gym.make(cfg.env.name) for _ in range(cfg.rl.num_envs)]
        )

    return env


@hydra.main(config_path="../cfg", config_name="learn_policy_tianshou.yaml")
def main(cfg):
    logger.info(f"Working directory: {os.getcwd()}")

    rl_cfg = cfg[cfg.rl.algorithm]
    train_env = make_env(cfg)
    env = train_env

    episodes = 10
    for _ in tqdm.tqdm(range(episodes), total=episodes):
        status = env.reset()
        while True:
            state, reward, done, _, _ = env.step([env.action_space[0].sample()])
            if done:
                break


if __name__ == "__main__":
    main()
