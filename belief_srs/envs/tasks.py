from abc import ABC, abstractmethod
from copy import deepcopy

from omegaconf import OmegaConf
import robosuite
from robosuite import load_controller_config
from .manip_search_wrapper import ManipSearchWrapper
from .improved_gym_wrapper import ImprovedGymWrapper
from .discrete_action_wrapper import DiscreteActionWrapper
from stable_baselines3.common.monitor import Monitor
from .rl_wrapper import RLWrapper
from .stack_v2 import StackV2


class Task(ABC):
    """Implements an abstract base Task class."""

    def __init__(self, cfg):
        """
        Args:
            env: test env
        """
        self.cfg = deepcopy(cfg)

    def create_env(self, gym=False, keys=None, eval=False, **kwargs):
        """
        Creates an new environment for the task.
        """
        if keys is None:
            keys = ['object-state',
                    'robot0_proprio-state']

        ctrl_cfg = OmegaConf.to_container(self.cfg.ctrl[self.cfg.controller])
        env_cfg = deepcopy(OmegaConf.to_container(self.cfg.env))

        if eval:
            env_cfg['has_offscreen_renderer'] = True
            env_cfg['use_camera_obs'] = True

        for key, val in kwargs.items():
            env_cfg[key] = val

        def _init():
            env = robosuite.make(**env_cfg, controller_configs=ctrl_cfg)
            env = ManipSearchWrapper(env, self.cfg)
            if gym:
                env = DiscreteActionWrapper(env, self.cfg.search.mprims)
                env = ImprovedGymWrapper(env, keys=keys)
                if eval:
                    env.render_mode = "rgb_array"
                env = Monitor(env)

            return env

        return _init

    def heuristic(self, state, heur_id=0, env=None):
        """
        Computes estimated cost-to-goal.

        Args:
            state: current state
            heur_id: 0 correponds to an admissible heuristic
        """
        if heur_id == 0:
            return self._heuristic_adm(state, env=env)
        else:
            return self._heuristic_inad(state, heur_id, env=env)

    @abstractmethod
    def _heuristic_adm(self, state, **kwargs):
        """Implements an admissible heuristic"""
        pass

    @abstractmethod
    def _heuristic_inad(self, state, heur_id, **kwargs):
        """Implements one or mode inadmissible heuristics"""
        pass


class BlockStackingTask(Task):
    """Implements a block stacking task."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.cfg.env.env_name = "Stack"
        self.cfg.env.env_name = "StackV2"

    def _heuristic_adm(self, state, env):
        h = env.reward_scale - env.reward(None)
        return h

    def _heuristic_inad(self, state, heur_id, env):
        return 0
