"""
Main script to train a policy for partially observable task using an
information gathering planner for guidance.
"""

from copy import deepcopy
import hydra
from hydra.utils import *
from functools import partial
import logging
import matplotlib.pyplot as plt
from minigrid.wrappers import (
    FullyObsWrapper,
    ImgObsWrapper,
    SymbolicObsWrapper,
)
import numpy as np
from omegaconf import OmegaConf
import os
from os.path import join
from PIL import Image
import ray
from skimage.transform import resize
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import stable_baselines3.common.callbacks as cb
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecMonitor,
    DummyVecEnv,
    VecVideoRecorder,
)
from stable_baselines3.common.torch_layers import NatureCNN
from time import sleep
import torch
from torchrl.record import VideoRecorder
from torchrl.envs import TransformedEnv, Compose
from tqdm import tqdm
import wandb
from wandb.integration.sb3 import WandbCallback

from belief_srs.envs.gridworld import POMDPGridWorld
from belief_srs.utils.wrappers import RGBImgObsWrapper
from belief_srs.utils.wandb import WandbLogger
from belief_srs.skills.utils import linear_schedule

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# torch.set_num_threads(4)


def make_env(cfg, is_test=False):
    render_mode = "human" if cfg.render else "rgb_array"
    device = torch.device(0) if torch.cuda.device_count() else torch.device("cpu")
    env_cfg = cfg.env[cfg.env.name]

    def _make_env(seed, rank=0):
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

            rl_cfg = cfg.rl[cfg.rl.algorithm]
            if cfg.rl.algorithm.endswith("torchrl"):
                from torchrl.envs.libs.gym import GymWrapper

                env = GymWrapper(env, device=device)

        else:
            raise NotImplementedError()

        return env

    rl_cfg = cfg.rl[cfg.rl.algorithm]
    if cfg.rl.algorithm.endswith("torchrl"):
        from torchrl.envs import ParallelEnv

        if cfg.rl.parallelize:
            env = ParallelEnv(
                cfg.rl.num_envs,
                _make_env,
                create_env_kwargs=[
                    {"seed": cfg.seed, "rank": i} for i in range(cfg.rl.num_envs)
                ],
            )
        else:
            env = _make_env(cfg.seed, rank=0)

            # if cfg.rl.record_video and not cfg.render:
            # from torchrl.envs import TransformedEnv
            # from torchrl.record import VideoRecorder

            # env = TransformedEnv(env, VideoRecorder(logger=wandb_logger, tag="dqn"))

    else:
        env = DummyVecEnv([lambda: _make_env(cfg.seed, rank=i) for i in range(1)])
        env = VecMonitor(env)

        if cfg.rl.record_video and not cfg.render:
            env = VecVideoRecorder(
                env,
                "./videos/",
                record_video_trigger=lambda x: x % cfg.rl.save_freq == 0,
                video_length=cfg.rl.video_length,
            )

    return env


def make_agent(cfg, env):
    rl_cfg = cfg.rl[cfg.rl.algorithm]

    if cfg.rl.algorithm == "dqn":
        from belief_srs.skills.dqn_agent import DQNAgent

        hyperparams = OmegaConf.to_container(rl_cfg.params)
        if rl_cfg.lr_schedule:
            hyperparams["learning_rate"] = linear_schedule(hyperparams["learning_rate"])
        policy_kwargs = dict(
            features_extractor_class=NatureCNN,
            # features_extractor_class=env.envs[0].get_feature_extractor()
            # if hasattr(env, "envs")
            # else env.get_feature_extractor(),
            features_extractor_kwargs=dict(features_dim=rl_cfg.policy.features_dim),
        )
        hyperparams["policy_kwargs"] = policy_kwargs
        agent = DQNAgent(env=env, **hyperparams)

    elif cfg.rl.algorithm == "recurrent_dqn":
        from belief_srs.skills.recurrent_dqn_agent import RecurrentDQNAgent

        hyperparams = OmegaConf.to_container(rl_cfg.params)
        if rl_cfg.lr_schedule:
            hyperparams["learning_rate"] = linear_schedule(hyperparams["learning_rate"])
        policy_kwargs = dict(
            features_extractor_class=NatureCNN,
            # features_extractor_class=env.envs[0].get_feature_extractor()
            # if hasattr(env, "envs")
            # else env.get_feature_extractor(),
            features_extractor_kwargs=dict(features_dim=rl_cfg.policy.features_dim),
        )
        policy_kwargs.update(OmegaConf.to_container(rl_cfg.policy))
        __import__("ipdb").set_trace()

        hyperparams["policy_kwargs"] = policy_kwargs
        agent = RecurrentDQNAgent(env=env, **hyperparams)

    elif cfg.rl.algorithm == "recurrent_dqn_torchrl":
        from belief_srs.skills.dqn_agent_torchrl import DQNAgentTorchrl

        agent = DQNAgentTorchrl(partial(make_env, cfg=cfg), rl_cfg, wandb_logger)

    elif cfg.rl.algorithm == "recurrent_ppo_torchrl":
        from belief_srs.skills.ppo_agent_torchrl import PPOAgentTorchrl

        agent = PPOAgentTorchrl(partial(make_env, cfg=cfg), rl_cfg, wandb_logger)

    elif cfg.rl.algorithm == "recurrent_sacd_torchrl":
        from belief_srs.skills.sacd_agent_torchrl import SACDAgentTorchrl

        agent = SACDAgentTorchrl(partial(make_env, cfg=cfg), rl_cfg, wandb_logger)

    else:
        raise NotImplementedError("Agent not implemented")

    return agent


def train_agent(cfg, env=None, agent=None):
    if not env:
        env = make_env(cfg)
    if not agent:
        agent = make_agent(cfg, env)
    rl_cfg = cfg.rl[cfg.rl.algorithm]

    ckpt_callback = cb.CheckpointCallback(
        save_freq=cfg.rl.save_freq,
        save_path="./policies",
        save_vecnormalize=True,
    )
    callbacks = [ckpt_callback]
    if cfg.use_wandb:
        wandb_callback = WandbCallback(
            model_save_path="policies/", model_save_freq=cfg.rl.save_freq, verbose=2
        )
        callbacks.append(wandb_callback)

    agent.learn(
        rl_cfg.total_timesteps,
        callback=callbacks,
    )


def evaluate_agent(cfg, env=None, agent=None):
    if not env:
        env = make_env(cfg)
    if not agent:
        agent = make_agent(cfg, env)
    rl_cfg = cfg.rl[cfg.rl.algorithm]
    agent.load(to_absolute_path(cfg.policy.path_to_policy))
    rews_mean, rews_std = evaluate_policy(
        agent, env, cfg.policy.n_eval_episodes, render=cfg.render
    )
    logger.info("Evaluation results:")
    logger.info("--------------------")
    logger.info(f"  Mean reward: {np.mean(rews_mean)}")


def init_experiment(cfg):
    if cfg.comment:
        with open("README.txt", "w") as f:
            f.write(cfg.comment)

    if cfg.use_ray:
        ray.init(num_cpus=cfg.num_cpus + 2, num_gpus=cfg.num_gpus)

    if cfg.use_wandb:
        config = {
            "env": OmegaConf.to_container(cfg.env[cfg.env.name], resolve=True),
            "rl": OmegaConf.to_container(cfg.rl[cfg.rl.algorithm], resolve=True),
        }
        # run = wandb.init(
        # project="belief_srs",
        # config=config,
        # # sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # # monitor_gym=True,  # auto-upload the videos of agents playing the game
        # # save_code=True,  # optional
        # )
        wandb_logger = WandbLogger(
            exp_name=cfg.tag if cfg.tag else None,
            project="belief_srs",
            config=config,
        )
        # make global
        globals()["wandb_logger"] = wandb_logger

    logger.info(OmegaConf.to_yaml(cfg))


@hydra.main(config_path="../cfg", config_name="learn_policy_with_planner.yaml")
def main(cfg):
    init_experiment(cfg)
    env = make_env(cfg)
    agent = make_agent(cfg, env)

    if cfg.policy.learn:
        train_agent(cfg, env, agent)

    if cfg.policy.evaluate:
        evaluate_agent(cfg, env, agent)

    if cfg.debug.viz_full_obs:
        logger.info("Visualizing full obs")
        env = TransformedEnv(
            env, Compose(VideoRecorder(logger=wandb_logger, tag="full_obs"))
        )
        if cfg.render is True:
            raise ValueError("Cannot render and save images at the same time")
        env.rollout(100)
        env.transform.dump(suffix="1")

        # for _ in range(200):
        # env.rollout(5)
        # env.transform.dump(suffix="2")

    if cfg.debug.viz_partial_view:
        logger.info("Visualizing partial view")
        if cfg.render is True:
            raise ValueError("Cannot render and save images at the same time")
        fig, ax = plt.subplots()
        for i in range(8):
            obs = env.reset()
            partial_view = obs[0]
            # ax.imshow(obs[0])
            full_view = env.render()

            scaled_partial_view = resize(partial_view, full_view.shape)
            combined_view = np.concatenate(
                (full_view, np.zeros((full_view.shape[0], 10, 3)), scaled_partial_view),
                axis=1,
            )
            ax.imshow(combined_view)
            fig.savefig(f"combined_view_{i}.png")


if __name__ == "__main__":
    main()
