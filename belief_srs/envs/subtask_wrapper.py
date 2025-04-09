from gymnasium.core import Env
import numpy as np
from robosuite.wrappers import Wrapper
from stable_baselines3.common.vec_env import VecEnv


class SubtaskWrapper(Wrapper, Env):
    """
    Defines a sub-task by:
        - a set of start states
        - a subgoal that defines
            - a reward function
            - a goal condition
    """

    def __init__(self, env, start_states, subgoal="task_goal"):
        super().__init__(env=env)
        self.start_states = start_states
        self.subgoal = subgoal
        self.subtask_reset = True

    def set_subtask_reset(self, value):
        self.subtask_reset = value

    def reset(self, **kwargs):
        if self.subtask_reset:
            start = np.random.choice(self.start_states)
            return self.reset_env(start)
        else:
            return super().reset(**kwargs)

    def reward(self, action=None):
        if self.subgoal is "task_goal":
            return self.env.reward(action)
        else:
            true_state = self.env.observe_true_state()
            # subgoal could be a learned model, hence, may not have access to
            # the env API
            return self.subgoal.reward(true_state, action)

    def set_start_states(self, states):
        self.start_states = states

    def set_subgoal(self, subgoal):
        self.subgoal = subgoal

    def step(self, *args, **kwargs):
        """To support 'render' arg."""
        return self.env.step(*args, **kwargs)

    def apply_skill(self, skill, obs, render=False):
        return skill.apply(self, obs, render)

    def reset_env(self, state):
        # needed to reset wrappers
        self.env.reset()
        mj_xml = state["mj_xml"]
        mj_xml = mj_xml.replace("/home/aries", "/home/svats2")
        mj_xml = mj_xml.replace("anaconda3", "miniconda3")
        mj_xml = mj_xml.replace("hrl_manip", "belief_srs")
        mj_xml = mj_xml.replace("hrl", "belief")
        self.env.reset_from_xml_string(mj_xml)
        self.env.sim.set_state_from_flattened(state["mj_state"])
        self.env.sim.forward()

        if "target_pos" in state:
            self.env.set_target_pos(state["target_pos"])

        if "box_quat_start" in state:
            self.env.set_box_quat_start(state["box_quat_start"])

        if "time" in state["obs"]:
            try:
                self.env.set_time(state["obs"]["time"][0])
            except:
                pass

        obs = self.env._get_observations(force_update=True)
        return obs


def reset_env(env, state):
    # not needed as reset_from_xml calls reset
    # self.env.reset()

    mj_xml = state["mj_xml"]
    mj_xml = mj_xml.replace("/home/aries", "/home/svats2")
    mj_xml = mj_xml.replace("anaconda3", "miniconda3")
    # TODO make this efficient by only setting data and model
    env.reset_from_xml_string(mj_xml)
    env.sim.set_state_from_flattened(state["mj_state"])
    env.sim.forward()

    if "target_pos" in state:
        env.set_target_pos(state["target_pos"])

    if "box_quat_start" in state:
        env.set_box_quat_start(state["box_quat_start"])

    if "time" in state["obs"]:
        try:
            env.set_time(state["obs"]["time"][0])
        except:
            pass

    obs = env._get_observations(force_update=True)
    return obs
