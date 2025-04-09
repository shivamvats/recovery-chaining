import logging
from copy import deepcopy

import belief_srs.utils.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import stable_baselines3.common.callbacks as cb
# import ray
import torch
from belief_srs.utils import *
from gymnasium.spaces import Box, Dict, Discrete
from omegaconf import OmegaConf
from rl_utils.reps import Reps
from stable_baselines3 import SAC  # ,PPO
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecMonitor, VecNormalize,
                                              VecVideoRecorder)
from torch import nn
from tqdm import tqdm

from .ppo import PPO
from .ppo_precond import PPOPrecond
from .ppo_term_q import PPOTermQ
from .robot_skill import RobotSkill
from .utils import EvalCallback, WandbCallback

# from belief_srs.envs.rl_wrapper import RLWrapper

logger = logging.getLogger(__name__)
# logger.setLevel("INFO")
# logger.setLevel("DEBUG")


class RecoverySkillDRL:
    """
    Policy is trained using deep rl (PPO or SAC)
    """

    def __init__(self, cfg, mode=None):
        self.cfg = cfg
        self.rl_cfg = self.cfg.rl
        self.mode = mode
        self.model = None

    def apply(self, obs, render=False):
        """
        Args:
            env: base VecEnv
        """
        venv = self.vecnorm_env
        venv.training = False
        venv.norm_reward = False
        venv.norm_obs = True
        obs = venv.normalize_obs(obs)

        # should be done before calling apply
        # [env.reset_option_time() for env in venv.envs]

        # vecnomr env normalizes obs
        # option_t, done = 0, False
        # if isinstance(obs, dict):
        # # for obsx in obs:
        # # obsx["option_time"] = np.array([option_t])
        # obs = venv.normalize_obs(venv.envs[0]._flatten_obs(obs)).reshape(1, -1)
        # else:
        # # raise NotImplementedError

        done = False
        hist = {"state": []}
        while not done:
            action, states = self.model.predict(obs, deterministic=True)
            logger.debug(f"  DRL obs: {obs}")
            logger.debug(f"  DRL action: {action}")
            obs, rew, done, info = venv.step(action)
            hist["state"].extend(info[0]["hist"]["state"])
            # option_t += 1
            # whole skill counts as one action
            # for obsx in obs:
            # obsx["option_time"] = np.array([option_t])

        info[0]["hist"] = hist
        return obs[0], rew[0], done[0], info[0]

    def train_policy(self, train_env_fn, eval_env_fn, continue_training=False):
        import torch
        from belief_srs.envs.subtask_wrapper import reset_env

        device = torch.device(self.cfg.device)
        torch.set_grad_enabled(True)
        algorithm = self.rl_cfg.algorithm
        rl_cfg = deepcopy(self.rl_cfg[algorithm]["params"])

        if self.cfg.num_cpus == 1:
            train_env = VecNormalize(
                DummyVecEnv([train_env_fn for _ in range(self.cfg.num_cpus)]),
                norm_reward=self.rl_cfg.norm_reward,
                norm_obs=True,
            )
        else:
            train_env = VecNormalize(
                SubprocVecEnv(
                    [train_env_fn for _ in range(self.cfg.num_cpus)]),
                norm_reward=self.rl_cfg.norm_reward,
                norm_obs=True,
            )
        if algorithm == "PPOTermQ":
            num_cpus_rollout = self.rl_cfg[algorithm].critic.n_cpus_rollout
            if num_cpus_rollout < 0:
                num_cpus_rollout = self.cfg.num_cpus
        else:
            num_cpus_rollout = self.cfg.num_cpus

        if algorithm == "PPOTermQ" or self.rl_cfg.evaluate:
            if num_cpus_rollout == 1:
                rollout_env = VecNormalize(
                    DummyVecEnv(
                        [eval_env_fn for _ in range(self.cfg.num_cpus)]),
                    norm_reward=self.rl_cfg.norm_reward,
                    norm_obs=True,
                )
            else:
                rollout_env = VecNormalize(
                    SubprocVecEnv(
                        [eval_env_fn for _ in range(self.cfg.num_cpus)]),
                    norm_reward=self.rl_cfg.norm_reward,
                    norm_obs=True,
                )
            rollout_env = VecMonitor(rollout_env)
            eval_env = rollout_env

        train_env = VecMonitor(train_env)
        self.vec_norm_env = train_env

        # EvalCallback syncs train and eval VecNormalize
        # eval_env = VecNormalize(
        # DummyVecEnv([eval_env_fn]), training=False, norm_obs=True, norm_reward=False
        # )
        # eval_env = VecMonitor(eval_env)

        logger.info(f"  Training using {train_env.num_envs} envs")

        # eval_env1 = VecVideoRecorder(
        # eval_env,
        # "./videos_nondet/",
        # record_video_trigger=lambda x: x == 0,
        # video_length=self.cfg.env.horizon,
        # )
        eval_kwargs = OmegaConf.to_container(self.rl_cfg.eval)
        eval_kwargs["eval_freq"] = eval_kwargs["eval_freq"] // self.cfg.num_cpus
        eval_kwargs["min_train_timesteps"] = (
            eval_kwargs["min_train_timesteps"] // self.cfg.num_cpus
        )
        logger.info(
            f" Eval callback: eval_freq: {eval_kwargs['eval_freq']}, min_timesteps: {eval_kwargs['min_train_timesteps']}"
        )
        # eval_callback_nondet = EvalCallback(
        # eval_env=eval_env,
        # best_model_save_path="./policies/",
        # log_path="./policies/",
        # **eval_kwargs,
        # # deterministic=True,
        # deterministic=False,  # to actually see the entropy
        # render=False,
        # )
        if self.rl_cfg.evaluate:
            eval_callback_det = EvalCallback(
                eval_env=eval_env,
                best_model_save_path="./policies/",
                log_path="./policies/",
                **eval_kwargs,
                # deterministic=True,
                deterministic=True,  # to actually see the entropy
                render=False,
            )
        ckpt_callback = cb.CheckpointCallback(
            save_freq=eval_kwargs["eval_freq"],
            save_path="./policies",
            save_vecnormalize=True,
        )
        wandb_callback = WandbCallback(
            best_model_save_path="./policies/",
            log_path="./policies/",
        )
        policy_kwargs = OmegaConf.to_container(self.rl_cfg[algorithm].policy)
        rl_cfg = deepcopy(self.rl_cfg[algorithm]["params"])
        callbacks = [ckpt_callback, wandb_callback]
        if self.rl_cfg.evaluate:
            callbacks.append(eval_callback_det)
            logger.info("  Adding eval callback")

        if self.rl_cfg.algorithm == "PPOTermQ":
            rl_cfg["n_steps"] = max(rl_cfg["n_steps"] // self.cfg.num_cpus, 1)
            if self.model is None:
                rl_agent = PPOTermQ(
                    "MlpPolicy",
                    train_env,
                    **rl_cfg,
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    tensorboard_log=".",
                    device=device,
                    num_nominal_actions=train_env.get_attr(
                        "num_nominal_actions")[0],
                    cfg=self.cfg,
                    rollout_env=rollout_env,
                )
            else:
                rl_agent = self.model

            rl_agent.learn(
                total_timesteps=self.rl_cfg.learn.n_total_steps,
                # demo_trajs=
                callback=callbacks,
                tb_log_name="PPO",
                reset_num_timesteps=False,
                cfg=self.cfg,
            )
        elif self.rl_cfg.algorithm == "PPO":
            rl_cfg["n_steps"] = max(rl_cfg["n_steps"] // self.cfg.num_cpus, 1)

            if self.model is None:
                if self.rl_cfg.use_nominal_precond:
                    rl_agent = PPOPrecond(
                        "MlpPolicy",
                        train_env,
                        **rl_cfg,
                        policy_kwargs=policy_kwargs,
                        verbose=1,
                        tensorboard_log=".",
                        device=device,
                        cfg=self.cfg,
                    )
                else:
                    rl_agent = PPO(
                        "MlpPolicy",
                        train_env,
                        **rl_cfg,
                        policy_kwargs=policy_kwargs,
                        verbose=1,
                        tensorboard_log=".",
                        device=device,
                    )
            else:
                rl_agent = self.model

            rl_agent.learn(
                total_timesteps=self.rl_cfg.learn.n_total_steps,
                # callback=[ckpt_callback, eval_callback_nondet, eval_callback_det],
                callback=callbacks,
                tb_log_name="PPO",
                reset_num_timesteps=False,
            )

        elif self.rl_cfg.algorithm == "SAC":
            if self.model is None:
                rl_agent = SAC(
                    "MlpPolicy",
                    train_env,
                    **rl_cfg,
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    tensorboard_log=".",
                )
            else:
                rl_agent = self.model
            rl_agent.learn(
                total_timesteps=self.rl_cfg.learn.n_total_steps,
                # callback=[ckpt_callback, eval_callback_nondet, eval_callback_det],
                callback=[ckpt_callback, eval_callback_det],
                tb_log_name="SAC",
                reset_num_timesteps=False,
            )

        else:
            raise NotImplementedError

        self.model = rl_agent

    def load(self, path_to_policy, path_to_vnorm, env, **kwargs):
        """
        Load a saved RL agent.
        """
        # loads vecnormalize stats
        # TODO: don't load
        env = VecNormalize.load(path_to_vnorm, env)
        self.vecnorm_env = env
        device = torch.device(self.cfg.device)
        try:
            model = PPO.load(path_to_policy, env=None, device=device, **kwargs)
        except:
            model = SAC.load(path_to_policy, env=None, **kwargs)
        self.model = model

    def save(self):
        self.vec_norm_env.save("vec_normalized.pkl")


# class ShelfRecoverySkillREPS(RobotSkill):
class ShelfRecoverySkill(RobotSkill):
    """Open-loop block recovery skill defined by parameters."""

    def __init__(self, params):
        super().__init__(steps_per_action=50)
        self.params = params

    def apply(self, env, obs, render=False):
        obs, rew, done, info = self._apply_params(self.params, env, render)

        return obs, rew, done, info

    def train_policy(self, train_env_fn, eval_env_fn, cfg):
        train_env = train_env_fn()
        eval_env = eval_env_fn()

        rl_cfg = cfg.rl
        reps = Reps(
            rel_entropy_bound=rl_cfg.rel_entropy_bound,
            min_temperature=rl_cfg.min_temperature,
            policy_variance_model="standard",
        )
        # collect samples with a random policy
        # initialize with uniform random exploration
        expl_space_pos = Box(
            low=np.array([-0.05, -0.05, -0.05, -0.05, -0.05, -0.05]),
            high=np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
        )
        # expl_space_grip = Discrete(3, start=-1)
        # expl_space = Dict({"position": expl_space_pos, "gripper": expl_space_grip})
        expl_space = expl_space_pos
        param_samples = []
        for _ in range(rl_cfg.initial_rollouts):
            sample = expl_space.sample()
            param_samples.append(
                sample
                # np.concatenate([sample["position"], [sample["gripper"]]])
            )

        train_info = {"hist": {"rews": []}}

        if cfg.use_ray:
            rollouts = self._collect_rollouts_ray(
                train_env_fn, param_samples, cfg.render
            )
        else:
            rollouts = self._collect_rollouts(
                train_env, param_samples, cfg.render)

        train_info["hist"]["rews"].append(rollouts["rews"])
        self.analyze_rollouts(rollouts, train_info, "_0")

        params_mean, params_cov, reps_info = reps.policy_from_samples_and_rewards(
            rollouts["params"], rollouts["rews"]
        )

        converged = False
        for i in tqdm(range(rl_cfg.max_updates)):
            rollouts = self._collect_rollouts(
                train_env, param_samples, cfg.render)
            train_info["hist"]["rews"].append(rollouts["rews"])
            self.analyze_rollouts(rollouts, train_info, f"_{i}")
            logger.info(f"  Params mean: {params_mean}")
            logger.info(f"  Max cov: {np.max(params_cov)}")
            logger.info(
                f"  Rewards: mean: {np.mean(rollouts['rews'])}, std: {np.std(rollouts['rews'])}"
            )
            params_mean, params_cov, reps_info = reps.policy_from_samples_and_rewards(
                rollouts["params"], rollouts["rews"]
            )
            # evaluate
            if np.all(params_cov < (0.001 * 0.001)):
                logger.info("REPS Converged")
                logger.info(f"Average reward: {np.mean(rollouts['rews'])}")
                converged = True
                break

            # sample from Gaussian
            param_samples = np.random.multivariate_normal(
                params_mean, params_cov, size=rl_cfg.rollouts_per_update
            )
        self.params = params_mean

    def _collect_rollouts(self, env, param_samples, render=False):
        rollouts = {"params": [], "rews": []}
        for params in param_samples:
            obs = env.reset()
            obs, rew, done, info = self._apply_params(
                params, env, render, self.steps_per_action
            )
            rollouts["params"].append(params)
            rollouts["rews"].append(rew)
        pkl_dump(rollouts, "rollouts_0.pkl")
        return rollouts

    @staticmethod
    def _apply_params(params, env, render=False, steps_per_action=50):
        delta_x1, delta_y1, delta_z1, delta_x2, delta_y2, delta_z2 = params
        eef_pos, eef_mat = RobotSkill.eef_pose(env)

        target_pos = eef_pos + np.array([delta_x1, delta_y1, delta_z1])
        target_axisangle = T.mat2axisangle(eef_mat)
        action = np.concatenate([target_pos, target_axisangle, [1]])
        obs, rew, done, info = RobotSkill.apply_action(env, action, render)

        target_pos = eef_pos + np.array([delta_x2, delta_y2, delta_z2])
        target_axisangle = T.mat2axisangle(eef_mat)
        action = np.concatenate([target_pos, target_axisangle, [1]])
        obs, rew, done, info = RobotSkill.apply_action(env, action, render)

        # retract
        eef_pos, eef_mat = RobotSkill.eef_pose(env)
        eef_ori = T.quat2axisangle(T.mat2quat(eef_mat))
        target_pos = eef_pos
        action = np.concatenate([target_pos, eef_ori, [-1]])
        obs, rew, done, info = RobotSkill.apply_action(
            env, action, render, steps_per_action=steps_per_action
        )

        eef_pos, eef_mat = RobotSkill.eef_pose(env)
        target_pos[1] -= 0.1
        action = np.concatenate([target_pos, eef_ori, [-1]])
        obs, rew, done, info = RobotSkill.apply_action(
            env, action, render, steps_per_action=steps_per_action
        )
        return obs, rew, done, info

    @staticmethod
    def analyze_rollouts(rollouts, train_info, suffix=""):
        params = rollouts["params"]
        rews = rollouts["rews"]
        good_params = np.array([p for p, r in zip(params, rews) if r > 0])
        if len(good_params) > 0:
            fig, ax = plt.subplots(2)
            ax[0].set_title("position params")
            ax[0].set_ylim((-0.05, 0.05))
            n = len(good_params)
            ax[0].scatter(np.arange(n), good_params[:, 0],
                          color="r", label="x")
            ax[0].scatter(np.arange(n), good_params[:, 1],
                          color="g", label="y")
            ax[0].scatter(np.arange(n), good_params[:, 2],
                          color="b", label="z")
            ax[1].set_title("gripper params")
            ax[1].scatter(np.arange(n), good_params[:, 3],
                          color="k", label="g")
            plt.tight_layout()
            fig.legend()
            plt.savefig(f"rollout_analysis{suffix}.png")

            fig, ax = plt.subplots()
            rews_mean_hist = [np.mean(rews)
                              for rews in train_info["hist"]["rews"]]
            ax.bar(np.arange(len(rews_mean_hist)), rews_mean_hist)
            plt.savefig("rewards.png")
            plt.close("all")

        else:
            logger.warning("No good params found!")
