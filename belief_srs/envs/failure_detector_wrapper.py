import logging

import numpy as np
from gymnasium.core import Env
from robosuite.wrappers import Wrapper

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)


class DoorFailureDetector(Wrapper, Env):
    def __init__(self, env, cfg, failure_detection=True):
        self.cfg = cfg
        self.failure_detection = failure_detection
        super().__init__(env=env)
        self.n_fails = 0

    def step(self, action, *args, **kwargs):
        obs, rew, done, info = self.env.step(action, *args, **kwargs)
        # logger.info(f"#Failuires: {self.n_fails}")
        if self._failure_condition(obs):
            done = True
            info["is_failure"] = True
            info["state"] = self.observe_true_state(obs)
            rew += self.cfg.rl.reward.fail_reward
        else:
            info["is_failure"] = False

        if self.env._check_success():
            info["is_success"] = True
            done = True
        else:
            info["is_success"] = False

        return obs, rew, done, info

    def _failure_condition(self, obs):
        if self.failure_detection:
            # if np.max(np.abs(self.env.robots[0].ee_force)) > 250:
            if np.max(np.abs(self.env.robots[0].ee_force)) > 550:
                # print(np.max(np.abs(self.env.robots[0].ee_force)))
                logger.debug("  Collision")
                return True

            # gripper_width = (
                # obs["robot0_gripper_qpos"][0] - obs["robot0_gripper_qpos"][1]
            # )
            # if gripper_width <= 0.01:
                # logger.debug("  Handle missed!")
                # return True

        return False

    def observe_true_state(self, obs=None, mj_state=True):
        state = {}
        if obs is None:
            state["obs"] = self._get_observations(force_update=False)
        else:
            state["obs"] = obs
        if mj_state:
            state["mj_state"] = self.sim.get_state().flatten()
            state["mj_xml"] = self.sim.model.get_xml()

        return state


class PickPlaceFailureDetector(Wrapper, Env):
    """
    Checks for failures conditions and sets the done flag accordingly.
    """

    def __init__(self, env, cfg, failure_detection=True):
        self.cfg = cfg
        self.failure_detection = failure_detection
        super().__init__(env=env)
        self.n_fails = 0

    def step(self, action, *args, **kwargs):
        obs, rew, done, info = self.env.step(action, *args, **kwargs)
        # logger.info(f"#Failuires: {self.n_fails}")
        if self._failure_condition(obs):
            done = True
            info["is_failure"] = True
            info["state"] = self.observe_true_state(obs)
            rew += self.cfg.rl.reward.fail_reward
        else:
            info["is_failure"] = False

        if self.env._check_success():
            info["is_success"] = True
            done = True
        else:
            info["is_success"] = False

        return obs, rew, done, info

    def _failure_condition(self, obs):
        if self.failure_detection:
            if np.max(np.abs(self.env.robots[0].ee_force)) > 50:
                logger.debug("  Collision")
                return True

            gripper_width = (
                obs["robot0_gripper_qpos"][0] - obs["robot0_gripper_qpos"][1]
            )
            if gripper_width <= 0.01:
                logger.debug("  Object dropped!")
                return True

        return False

    def observe_true_state(self, obs=None, mj_state=True):
        state = {}
        if obs is None:
            state["obs"] = self._get_observations(force_update=False)
        else:
            state["obs"] = obs
        if mj_state:
            state["mj_state"] = self.sim.get_state().flatten()
            state["mj_xml"] = self.sim.model.get_xml()

        return state


class PickFailureDetector(Wrapper, Env):
    """
    Checks for failures conditions and sets the done flag accordingly.

    Failure conditions:
    1. Collision
    2. Object dropped
    """

    def __init__(self, env, cfg, failure_detection=True):
        self.cfg = cfg
        self.failure_detection = failure_detection
        super().__init__(env=env)

    def step(self, action, *args, **kwargs):
        obs, rew, done, info = self.env.step(action, *args, **kwargs)
        if self._failure_condition(obs):
            done = True
            info["is_failure"] = True
            info["state"] = self.observe_true_state(obs)
            rew += self.cfg.rl.reward.fail_reward
        else:
            info["is_failure"] = False

        if self.env._check_success():
            info["is_success"] = True
            done = True
        else:
            info["is_success"] = False

        return obs, rew, done, info

    def _failure_condition(self, obs):
        if self.failure_detection:
            if np.max(np.abs(self.env.robots[0].ee_force)) > 50:
                logger.debug("  Collision")
                return True

            gripper_width = (
                obs["robot0_gripper_qpos"][0] - obs["robot0_gripper_qpos"][1]
            )
            if gripper_width <= 0.01:
                logger.debug("  Object dropped!")
                return True

            # TODO
            # check object height

        return False

    def observe_true_state(self, obs=None, mj_state=True):
        state = {}
        if obs is None:
            state["obs"] = self._get_observations(force_update=False)
        else:
            state["obs"] = obs
        if mj_state:
            state["mj_state"] = self.sim.get_state().flatten()
            state["mj_xml"] = self.sim.model.get_xml()

        return state
