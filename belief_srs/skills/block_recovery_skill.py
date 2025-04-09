import logging
import numpy as np
import belief_srs.utils.transforms as T
from gymnasium.spaces import Box, Discrete, Dict
import matplotlib.pyplot as plt
from tqdm import tqdm

from .robot_skill import RobotSkill
from rl_utils.reps import Reps
from belief_srs.utils import *

logger = logging.getLogger(__name__)


class BlockRecoverySkill(RobotSkill):
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
            low=np.array([-0.05, -0.05, -0.05]), high=np.array([0.05, 0.05, 0.05])
        )
        expl_space_grip = Discrete(3, start=-1)
        expl_space = Dict({"position": expl_space_pos, "gripper": expl_space_grip})
        param_samples = []
        for _ in range(rl_cfg.initial_rollouts):
            sample = expl_space.sample()
            param_samples.append(
                np.concatenate([sample["position"], [sample["gripper"]]])
            )

        train_info = {'hist': {'rews': []}}

        rollouts = self._collect_rollouts(train_env, param_samples, cfg.render)
        train_info['hist']['rews'].append(rollouts['rews'])
        self.analyze_rollouts(rollouts, train_info, "_0")

        params_mean, params_cov, reps_info = reps.policy_from_samples_and_rewards(
            rollouts["params"], rollouts["rews"]
        )

        converged = False
        for i in tqdm(range(rl_cfg.max_updates)):
            rollouts = self._collect_rollouts(train_env, param_samples, cfg.render)
            train_info['hist']['rews'].append(rollouts['rews'])
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
                logger.info("RESP Converged")
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
            obs, rew, done, info = self._apply_params(params, env, render)
            rollouts["params"].append(params)
            rollouts["rews"].append(rew)
        pkl_dump(rollouts, "rollouts_0.pkl")
        return rollouts

    def _apply_params(self, params, env, render=False):
        delta_x, delta_y, delta_z, gripper = params
        eef_pos, eef_mat = self.eef_pose(env)

        target_pos = eef_pos + np.array([delta_x, delta_y, delta_z])
        target_axisangle = T.mat2axisangle(eef_mat)
        target_gripper = gripper
        action = np.concatenate([target_pos, target_axisangle, [-1]])
        obs, rew, done, info = self.apply_action(env, action, render)

        action = np.concatenate([target_pos, target_axisangle, [target_gripper]])
        obs, rew, done, info = self.apply_action(env, action, render)

        # open gripper
        action = np.concatenate([target_pos, target_axisangle, [-1]])
        obs, rew, done, info = self.apply_action(env, action, render)

        # retract
        eef_pos, eef_mat = self.eef_pose(env)
        eef_ori = T.quat2axisangle(T.mat2quat(eef_mat))
        target_pos = eef_pos
        target_pos[2] += 0.1
        action = np.concatenate([target_pos, eef_ori, [-1]])
        obs, rew, done, info = self.apply_action(env, action, render)
        return obs, rew, done, info

    @staticmethod
    def analyze_rollouts(rollouts, train_info, suffix=""):
        params = rollouts["params"]
        rews = rollouts["rews"]
        good_params = np.array([p for p, r in zip(params, rews) if r > 0])
        fig, ax = plt.subplots(2)
        ax[0].set_title("position params")
        ax[0].set_ylim((-0.05, 0.05))
        n = len(good_params)
        ax[0].scatter(np.arange(n), good_params[:, 0], color="r", label="x")
        ax[0].scatter(np.arange(n), good_params[:, 1], color="g", label="y")
        ax[0].scatter(np.arange(n), good_params[:, 2], color="b", label="z")
        ax[1].set_title("gripper params")
        ax[1].scatter(np.arange(n), good_params[:, 3], color="k", label="g")
        plt.tight_layout()
        fig.legend()
        plt.savefig(f"rollout_analysis{suffix}.png")

        fig, ax = plt.subplots()
        rews_mean_hist = [np.mean(rews) for rews in train_info["hist"]["rews"]]
        ax.bar(np.arange(len(rews_mean_hist)), rews_mean_hist)
        plt.savefig("rewards.png")
        plt.close('all')


