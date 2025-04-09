from copy import deepcopy
import logging
import numpy as np
from .robot_skill import RobotSkill
from belief_srs.utils.transforms import *

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


class PrimitiveSkill(RobotSkill):
    """
    Implements a motion primitive defined in terms of relative eef motion.

    Args:
        delta_pos: relative eef motion
        delta_rpy: relative rpy motion in the eef frame
        gripper: gripper action
        timesteps: simulation steps
    """

    def __init__(self, delta_pos, delta_rpy, gripper, timesteps):
        self.action_pos = delta_pos
        self.action_rpy = delta_rpy
        self.action_gripper = gripper
        self.timesteps = timesteps
        self.discount = 0.99

    def apply(self, env, obs, render=False):
        # doesn't reset option_time
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

    def _apply(self, env, obs=None, render=False):
        curr_pos = env.sim.data.site_xpos[
            env.sim.model.site_name2id("gripper0_grip_site")
        ]
        curr_mat = np.array(
            env.sim.data.site_xmat[
                env.sim.model.site_name2id("gripper0_grip_site")
            ].reshape([3, 3])
        )
        curr_rpy = mat2euler(curr_mat)

        goal_pos = curr_pos + self.action_pos
        # TODO limit to [-pi, pi]
        goal_rpy = curr_rpy + self.action_rpy
        rpy_to_mat = RT.rotation_from_quaternion(
            T.convert_quat(
                AT.quaternion_from_euler(goal_rpy[0], goal_rpy[1], goal_rpy[2]), "wxyz"
            )
        )
        goal_T = RT(rpy_to_mat)
        goal_axis = goal_T.axis_angle
        action = np.concatenate([goal_pos, goal_axis, [self.action_gripper]])

        hist = {"reward": [], "state": []}
        t = 0
        for _ in range(self.timesteps):
            obs, rew, done, info = env.step(action)
            t += 1
            logger.debug(f"  option time: {obs['option_time']}")
            hist['reward'].append(rew)
            hist['state'].append({'obs': obs})
            if render:
                env.render()
            if done:
                break
        info["interval"] = t
        # info["hist"] = {"reward": rew_hist,
                        # "state":}
        info['hist'] = hist
        info["action"] = action
        info["action_cost"] = np.linalg.norm(
            np.linalg.norm(self.action_pos)
        ) + np.linalg.norm(self.action_rpy)

        return obs, rew, done, info


class PrimitiveAction(RobotSkill):
    """
    Implements a motion primitive defined in terms of an absolute pose.

    Args:
        target_pos: target position
        target_rpy: target rpy
        gripper: gripper action
        timesteps: simulation steps
    """

    def __init__(self, action, timesteps):
        self.action = action
        self.timesteps = timesteps

    def apply(self, env, noise=None, obs=None, render=False):
        if noise is None:
            noise = np.zeros_like(self.action)
        t = 0
        for _ in range(self.timesteps):
            obs, rew, done, info = env.step(
                self.action + noise)
            t += 1
            if render:
                env.render()
        info["interval"] = t
        return obs, rew, done, info
