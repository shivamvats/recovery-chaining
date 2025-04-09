"""
This file implements a wrapper for facilitating compatibility with OpenAI gym.
This is useful when using these environments with code that assumes a gym-like
interface.
"""

from copy import deepcopy
import numpy as np
import gymnasium as gym
from gymnasium import spaces, Env

from robosuite.wrappers import Wrapper


class ImprovedGymWrapper(Wrapper, gym.Env):
    metadata = None
    render_mode = "rgb_array"
    """
    Improves upon the robosuite GymWrapper in the following ways:
        1. supports discrete action space.

    Args:
        env (MujocoEnv): The environment to wrap.
        keys (None or list of str): If provided, each observation will
            consist of concatenated keys from the wrapped environment's
            observation dictionary. Defaults to proprio-state and object-state.

    Raises:
        AssertionError: [Object observations must be enabled if no keys]
    """

    def __init__(self, env, keys=None):
        # Run super method
        super().__init__(env=env)
        # Create name for gym
        robots = "".join(
            [type(robot.robot_model).__name__ for robot in self.env.robots]
        )
        self.name = robots + "_" + type(self.env).__name__

        # Get reward range
        self.reward_range = (0, self.env.reward_scale)

        if keys is None:
            keys = []
            # Add object obs if requested
            if self.env.use_object_obs:
                keys += ["object-state"]
            # Add image obs if requested
            if self.env.use_camera_obs:
                keys += [f"{cam_name}_image" for cam_name in self.env.camera_names]
            # Iterate over all robots to add to state
            for idx in range(len(self.env.robots)):
                keys += ["robot{}_proprio-state".format(idx)]
        self.keys = keys

        # Gym specific attributes
        self.env.spec = None

        # set up observation and action spaces
        self._start_eef_pos = np.zeros(3)
        obs = self.env.reset()
        print(obs.keys())
        self.modality_dims = {key: obs[key].shape for key in self.keys}
        flat_ob = self._flatten_obs(obs)
        self.obs_dim = flat_ob.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)
        if isinstance(self.env.action_spec, int):
            n_actions = self.env.action_spec
            self.action_space = spaces.Discrete(n_actions)
        else:
            low, high = self.env.action_spec
            self.action_space = spaces.Box(low, high)

        self.metadata = {"render.modes": ["human", "rgb_array"]}

    def terminate(self):
        self.terminated_ep = True

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed

        Returns:
            np.array: observations flattened into a 1d array
        """
        ob_lst = []
        for key in self.keys:
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                if key == "robot0_eef_pos":
                    rel_pos = obs_dict["robot0_eef_pos"] - self._start_eef_pos
                    ob_lst.append(rel_pos.flatten())
                else:
                    ob_lst.append(np.array(obs_dict[key]).flatten())
        return np.concatenate(ob_lst)

    def reset(self, seed=None, options=None):
        """
        Extends env reset method to return flattened observation instead of normal OrderedDict and optionally resets seed

        Returns:
            np.array: Flattened environment observation space after reset occurs
        """
        self.terminated_ep = False
        if seed is not None:
            if isinstance(seed, int):
                np.random.seed(seed)
            else:
                raise TypeError("Seed must be an integer type!")
        ob_dict = self.env.reset()
        return self._flatten_obs(ob_dict), {}

    def step(self, action, *args, **kwargs):
        """
        Extends vanilla step() function call to return flattened observation instead of normal OrderedDict.

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (np.array) flattened observations from the environment
                - (float) reward from the environment
                - (bool) episode ending after reaching an env terminal state
                - (bool) episode ending after an externally defined condition
                - (dict) misc information
        """
        if self.terminated_ep:
            print("Terminated episode")
            print("===================")
            return self._flatten_obs(self.env._get_observations()), 0, True, False, {"terminated": True, "timesteps": 0}

        if self.render_mode == "human":
            kwargs["render"] = True
            ob_dict, reward, terminated, info = self.env.step(action, *args, **kwargs)
            self.env.render()
        else:
            ob_dict, reward, terminated, info = self.env.step(action, *args, **kwargs)
        # print("  time: ", ob_dict["time"], "option time: ", ob_dict["option_time"])
        return self._flatten_obs(ob_dict), reward, terminated, False, info

    def render(self, *args, **kwargs):
        """
        By default, run the normal environment render() function

        Args:
            **kwargs (dict): Any args to pass to environment render function
        """
        if self.env.viewer is not None and self.render_mode == "human":
            self.env.render()

        if self.render_mode == "rgb_array" and self.env.use_camera_obs:
            # offscreen rendering
            return self.env._observables[
                f"{self.env.unwrapped.render_camera}_image"
            ].obs[::-1]

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward

        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]

        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        return self.env.reward()

    def set_start_eef_pos(self, pos):
        self._start_eef_pos = deepcopy(pos)

    def _get_observations(self, **kwargs):
        ob_dict = self.env._get_observations(**kwargs)
        return self._flatten_obs(ob_dict)
