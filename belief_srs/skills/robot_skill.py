from copy import deepcopy
import logging
import numpy as np

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


class RobotSkill:
    """Base class for all robot skills."""

    def __init__(self, steps_per_action=20, discount=0.99):
        self.steps_per_action = steps_per_action
        self.discount = discount
        self.time_per_action = 1.25

    def apply(self, env, obs, render=False):
        if hasattr(env, "reset_option_time"):
            env.reset_option_time()
        obs = deepcopy(obs)
        obs["option_time"] = np.array([0])
        obs, rew, done, info = self._apply(env, obs, render)
        # discounted reward for this option
        total_rew = np.sum(
            [rew * self.discount**i for i, rew in enumerate(info["hist"]["reward"])]
        )
        # length of option
        info['interval'] = len(info['hist']['reward'])

        logger.debug(f"  rew hist: {info['hist']['reward']}")
        logger.debug(f"    discounted rew: {total_rew}")

        return obs, total_rew, done, info

    def _apply(self, env, obs, render):
        raise NotImplementedError

    @staticmethod
    def apply_action(env, action, render=False, steps_per_action=20):
        hist = {"state": [], "reward": []}
        discount = 1.0  # 0.99
        option_rew = 0
        for _ in range(steps_per_action):
            # env.increment_time()
            obs, rew, done, info = env.step(action)
            state = env.observe_true_state()
            option_rew += discount * rew
            hist["state"].append(state)
            hist["reward"].append(rew)
            if render:
                env.render()
            if done:
                break

        info["hist"] = hist
        info["option"] = dict(reward=option_rew, state=hist["state"][-1])
        rew = np.sum(hist["reward"])
        return obs, rew, done, info

    # @staticmethod
    # def update_state_obs(env, obs, option_time=None):
    # true_state = env.observe_true_state(mj_state=True)
    # # if option_time is not None:
    # # obs['option_time'] = np.array([option_time])
    # # true_state['option_time'] = np.array([option_time])
    # return true_state, obs

    @staticmethod
    def eef_pose(env):
        eef_pos = np.array(
            env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")]
        )
        eef_mat = np.array(
            env.sim.data.site_xmat[env.sim.model.site_name2id("gripper0_grip_site")]
        ).reshape((3, 3))

        return eef_pos, eef_mat
