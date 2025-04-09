from gymnasium.core import Env
import numpy as np
from robosuite.wrappers import Wrapper
from stable_baselines3.common.vec_env import VecEnvWrapper


class PrecondVecEnvWrapper(VecEnvWrapper):
    def __init__(self, venv):
        self.precond = None
        super().__init__(venv)



