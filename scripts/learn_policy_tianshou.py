"""
Based on https://github.com/thu-ml/tianshou/blob/master/examples/atari/atari_sac.py
"""

import argparse
import datetime
import logging
import os
import pprint

import gymnasium as gym
from gymnasium.experimental.wrappers.rendering import RecordVideoV0 as RecordVideo
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

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.policy import SACPolicy, DiscreteSACPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.env.venvs import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import BasePolicy
from tianshou.trainer.utils import gather_info, test_episode
from tianshou.utils import (
    BaseLogger,
    DummyTqdm,
    LazyLogger,
    MovAvg,
    deprecation,
    tqdm_config,
)


import stable_baselines3.common.callbacks as cb
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecMonitor,
    DummyVecEnv,
    VecVideoRecorder,
)
from belief_srs.envs.gridworld import POMDPGridWorld
from belief_srs.utils.wrappers import RGBImgObsWrapper, ProcessImageWrapper
from belief_srs.utils.layers import NatureCNN, AtariDQN
from belief_srs.utils.tianshou_wandb import WandbLogger
import wandb

logger = logging.getLogger(__name__)

torch.set_num_threads(5)


def make_env(cfg, test=False):
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

            if test:
                env = RecordVideo(
                    env,
                    video_folder="video_test/",
                    episode_trigger=lambda x: True,
                    video_length=128*5,
                    # name_prefix="test"
                    disable_logger=True,
                )
            else:
                env = RecordVideo(
                    env,
                    video_folder="video_train/",
                    step_trigger=lambda x: x % cfg.rl.save_freq == 0,
                    video_length=128*5,
                    disable_logger=True,
                    name_prefix=f"env_{rank}"
                )

        else:
            raise NotImplementedError()

        return env

    if test:
        num_envs = cfg[cfg.rl.algorithm].test_envs
    else:
        num_envs = cfg[cfg.rl.algorithm].train_envs

    if cfg.env.name == "gridworld":
        if num_envs == 1:
            env = DummyVectorEnv(
                [
                    lambda: _make_env(cfg.seed, rank=i, test=test)
                    for i in range(num_envs)
                ]
            )
        else:
            env = SubprocVectorEnv(
                [
                    lambda: _make_env(cfg.seed, rank=i, test=test)
                    for i in range(num_envs)
                ]
            )
    else:
        env = DummyVectorEnv([lambda: gym.make(cfg.env.name) for _ in range(num_envs)])
    # env = VecMonitor(env)

    return env


def get_sweep_config():
    sweep_config = {
        "name": "rl_sweep",
        "method": "random",
        "metric": {"goal": "maximize", "name": "test/reward"},
        "parameters": {
            "sacd": {
                "buffer_size": {"values": [100_000]},
                "hidden_sizes": {"values": [[512, 512]]},
                "features_dim": {"values": [512]},
                "actor_lr": {"values": [1e-3]},
                "critic_lr": {"values": [1e-3]},
                "gamma": {"values": [0.99]},
                "tau": {"values": [0.005]},
                "alpha": {"values": [0.2]},
                "auto_alpha": {"values": [True]},
                "alpha_lr": {"values": [3e-4]},
                "start_timesteps": {"values": [10000]},
                "epoch": {"values": [100]},
                "step_per_epoch": {"values": [5000]},
                "step_per_collect": {"values": [1]},
                "update_per_step": {"values": [1]},
                "n_step": {"values": [1]},
                "batch_size": {"values": [128]},
                "training_num": {"values": [1]},
                "test_num": {"values": [10]},
            }
        },
    }
    return sweep_config


class OffpolicyTrainer1(OffpolicyTrainer):
    def train_step(self):
        """Perform one training step."""
        assert self.episode_per_test is not None
        assert self.train_collector is not None
        stop_fn_flag = False
        if self.train_fn:
            self.train_fn(self.epoch, self.env_step)
        result = self.train_collector.collect(
            n_step=self.step_per_collect, n_episode=self.episode_per_collect
        )
        if result["n/ep"] > 0 and self.reward_metric:
            rew = self.reward_metric(result["rews"])
            result.update(rews=rew, rew=rew.mean(), rew_std=rew.std())
        self.env_step += int(result["n/st"])
        self.logger.log_train_data(result, self.env_step)
        self.last_rew = result["rew"] if result["n/ep"] > 0 else self.last_rew
        self.last_len = result["len"] if result["n/ep"] > 0 else self.last_len
        data = {
            "env_step": str(self.env_step),
            "rew": f"{self.last_rew:.2f}",
            "len": str(int(self.last_len)),
            "n/ep": str(int(result["n/ep"])),
            "n/st": str(int(result["n/st"])),
        }
        if result["n/ep"] > 0:
            if self.test_in_train and self.stop_fn and self.stop_fn(result["rew"]):
                assert self.test_collector is not None
                test_result = test_episode(
                    self.policy,
                    self.test_collector,
                    self.test_fn,
                    self.epoch,
                    self.episode_per_test,
                    self.logger,
                    self.env_step,
                )
                if self.stop_fn(test_result["rew"]):
                    stop_fn_flag = True
                    self.best_reward = test_result["rew"]
                    self.best_reward_std = test_result["rew_std"]
                else:
                    self.policy.train()

        return data, result, stop_fn_flag

    # def test_step(self):
    # Retusuper().test_step()
    # self.make_video()

    def make_video(self):
        self.policy.eval()
        rec_env = RecordVideo(
            self.test_collector.env,
            video_folder="video/",
            episode_trigger=lambda x: True,
        )
        # rec_env.seed(args.seed)
        collector = Collector(self.policy, rec_env, exploration_noise=True)
        collector.collect(n_episode=5)


@hydra.main(config_path="../cfg", config_name="learn_policy_tianshou.yaml")
def main(cfg):
    logger.info(f"Working directory: {os.getcwd()}")

    rl_cfg = cfg[cfg.rl.algorithm]
    train_env = make_env(cfg)
    env = train_env
    test_env = make_env(cfg, test=True)

    state_shape = env.observation_space[0].shape or env.observation_space[0].n
    action_shape = env.action_space[0].shape or env.action_space[0].n
    print("Observations shape:", state_shape)
    print("Actions shape:", action_shape)
    # seed
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    # model
    # hidden_sizes = rl_cfg.hidden_layers * [rl_cfg.hidden_size]
    hidden_sizes = rl_cfg.hidden_sizes

    if rl_cfg.feature_net == "NatureCNN":
        feature_net_actor = NatureCNN(
            observation_space=env.observation_space[0],
            features_dim=rl_cfg.features_dim,
            normalized_image=True,
            device=cfg.device,
        ).to(cfg.device)
        feature_net_critic = NatureCNN(
            observation_space=env.observation_space[0],
            features_dim=rl_cfg.features_dim,
            normalized_image=True,
            device=cfg.device,
        ).to(cfg.device)

    elif rl_cfg.feature_net == "AtariDQN":
        feature_net_actor = AtariDQN(
            *state_shape,
            action_shape,
            device=cfg.device,
            features_only=True,
            output_dim=rl_cfg.features_dim,
        )
        if rl_cfg.share_features:
            feature_net_critic = feature_net_actor
        else:
            feature_net_critic = AtariDQN(
                *state_shape,
                action_shape,
                device=cfg.device,
                features_only=True,
                output_dim=rl_cfg.features_dim,
            )

    else:
        raise NotImplementedError
    actor = Actor(
        feature_net_actor,
        action_shape,
        preprocess_net_output_dim=feature_net_actor.features_dim,
        hidden_sizes=hidden_sizes,
        device=cfg.device,
        softmax_output=False,  # SAC will normalize
    ).to(cfg.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=rl_cfg.actor_lr)

    critic1 = Critic(
        feature_net_critic,
        hidden_sizes=hidden_sizes,
        preprocess_net_output_dim=feature_net_critic.features_dim,
        last_size=action_shape,
        device=cfg.device,
    ).to(cfg.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=rl_cfg.critic_lr)

    critic2 = Critic(
        feature_net_critic,
        hidden_sizes=hidden_sizes,
        preprocess_net_output_dim=feature_net_critic.features_dim,
        last_size=action_shape,
        device=cfg.device,
    ).to(cfg.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=rl_cfg.critic_lr)

    logger.info(f"Actor: {actor}")
    logger.info(f"Critic: {critic1}")

    if rl_cfg.auto_alpha:
        # target_entropy = -np.log(np.prod(action_shape))
        target_entropy = -np.prod(action_shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=cfg.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=rl_cfg.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = rl_cfg.alpha

    policy = DiscreteSACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic1=critic1,
        critic1_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=rl_cfg.tau,
        gamma=rl_cfg.gamma,
        alpha=alpha,
        estimation_step=rl_cfg.n_step,
        action_space=env.action_space,
        deterministic_eval=False,
    ).to(cfg.device)

    # load a previous policy
    if cfg.resume_path:
        policy.load_state_dict(torch.load(cfg.resume_path, map_location=cfg.device))
        print("Loaded agent from: ", cfg.resume_path)

    # collector
    buffer: VectorReplayBuffer | ReplayBuffer
    buffer = VectorReplayBuffer(rl_cfg.buffer_size, len(train_env))
    train_collector = Collector(policy, train_env, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_env)
    train_collector.collect(n_step=rl_cfg.start_timesteps, random=True)

    # log
    # now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    # cfg.algo_name = "sac"
    # log_name = os.path.join(cfg.task, cfg.algo_name, str(cfg.seed), now)
    # log_path = os.path.join(cfg.logdir, log_name)

    # logger
    wb_logger = WandbLogger(
        project=cfg.wandb_project,
        name=cfg.tag,
        config=OmegaConf.to_container(cfg, resolve=True),
        group=cfg.group,
        train_interval=1,
        test_interval=1,
        update_interval=1,
    )
    wb_logger.load(SummaryWriter("log"))

    def save_best_fn(policy: BasePolicy) -> None:
        torch.save(policy.state_dict(), "policy.pth")

    # trainer
    result = OffpolicyTrainer1(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=rl_cfg.epoch,
        step_per_epoch=rl_cfg.step_per_epoch,
        step_per_collect=rl_cfg.step_per_collect,
        update_per_step=rl_cfg.update_per_step,
        episode_per_test=rl_cfg.test_num,
        batch_size=rl_cfg.batch_size,
        save_best_fn=save_best_fn,
        logger=wb_logger,
        test_in_train=False,
    ).run()
    pprint.pprint(result)

    # Let's watch its performance!
    policy.eval()
    test_env.seed(cfg.seed)
    test_collector.reset()
    collector_stats = test_collector.collect(
        n_episode=rl_cfg.test_num, render=cfg.render
    )
    print(collector_stats)


if __name__ == "__main__":
    main()
