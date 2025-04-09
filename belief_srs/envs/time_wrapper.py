from gymnasium.core import Env
import numpy as np
from robosuite.wrappers import Wrapper


class TimeWrapper(Wrapper, Env):
    def __init__(self, env):
        super().__init__(env)
        self.t = 0
        self.option_t = 0

    def reset(self):
        obs = self.env.reset()
        self._reset_internal()
        return self._add_time(obs)

    def _reset_internal(self):
        self.t = 0
        self.option_t = 0

    def reset_option_time(self):
        self.option_t = 0

    def increment_time(self):
        self.t += 1
        self.option_t += 1

    def set_time(self, time):
        self.t = time

    def get_time(self):
        return self.t

    def step(self, action, increment_time=True, **kwargs):
        obs, rew, done, info = self.env.step(action, **kwargs)
        if increment_time:
            self.increment_time()
        obs = self._add_time(obs)
        # discrete wrapper handles horizon as I enforce this only on the learned skill
        return obs, rew, done, info

    def _get_observations(self, **kwargs):
        obs = self.env._get_observations(**kwargs)
        return self._add_time(obs)

    def observe_true_state(self, *args, **kwargs):
        if hasattr(self.env, "observe_true_state"):
            return self._add_time(self.env.observe_true_state(*args, **kwargs))
        else:
            return self._get_observations()

    def _add_time(self, obs):
        # tracks overall time
        obs["time"] = np.array([self.t])
        # handle semi-markov nature of options
        obs["option_time"] = np.array([self.option_t])
        return obs

