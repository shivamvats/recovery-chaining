import logging
from typing import Optional, Sequence
import numpy as np

from d3rlpy.dataset import (
    EpisodeBase,
    ReplayBuffer,
)
from d3rlpy.interface import QLearningAlgoProtocol
from d3rlpy.metrics.evaluators import WINDOW_SIZE, EvaluatorProtocol, make_batches

logger = logging.getLogger(__name__)


class ReturnErrorEvaluator(EvaluatorProtocol):
    r"""Returns average error vs the returns

    Args:
        episodes: Optional evaluation episodes. If it's not given, dataset
            used in training will be used.
    """
    _episodes: Optional[Sequence[EpisodeBase]]

    def __init__(self, episodes: Optional[Sequence[EpisodeBase]] = None):
        self._episodes = episodes

    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBuffer,
    ) -> float:
        total_errors = []
        total_returns = []
        episodes = self._episodes if self._episodes else dataset.episodes
        for episode in episodes:
            values = algo.predict_value(episode.observations, episode.actions)
            returns = self._compute_returns(episode.rewards, algo.gamma)
            total_errors += ((values - returns) ** 2).tolist()
            # total_returns += returns.tolist()
        # print("Avg returns: ", np.mean(total_returns))
        # __import__('ipdb').set_trace()
        return np.sqrt(np.mean(total_errors))

    def _compute_returns(self, rewards, discount=0.99):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + discount * R
            returns.insert(0, R)
        return np.array(returns).flatten()


class ImprovedTDErrorEvaluator(EvaluatorProtocol):
    r"""Returns average TD error.

    This metric suggests how Q functions overfit to training sets.
    If the TD error is large, the Q functions are overfitting.

    .. math::

        \mathbb{E}_{s_t, a_t, r_{t+1}, s_{t+1} \sim D}
            [(Q_\theta (s_t, a_t)
             - r_{t+1} - \gamma \max_a Q_\theta (s_{t+1}, a))^2]

    Args:
        episodes: Optional evaluation episodes. If it's not given, dataset
            used in training will be used.
    """
    _episodes: Optional[Sequence[EpisodeBase]]

    def __init__(
        self, episodes: Optional[Sequence[EpisodeBase]] = None, value_clip_range=None
    ):
        self._episodes = episodes
        self._value_clip_range = value_clip_range

    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBuffer,
    ) -> float:
        total_errors = []
        episodes = self._episodes if self._episodes else dataset.episodes
        for episode in episodes:
            for batch in make_batches(episode, WINDOW_SIZE, dataset.transition_picker):
                # estimate values for current observations
                values = algo.predict_value(batch.observations, batch.actions)
                if self._value_clip_range:
                    values = np.clip(values, *self._value_clip_range)

                # estimate values for next observations
                next_actions = algo.predict(batch.next_observations)
                next_values = algo.predict_value(batch.next_observations, next_actions)

                if self._value_clip_range:
                    next_values = np.clip(next_values, *self._value_clip_range)

                # calculate td errors
                mask = (1.0 - batch.terminals).reshape(-1)
                rewards = np.asarray(batch.rewards).reshape(-1)
                if algo.reward_scaler:
                    rewards = algo.reward_scaler.transform_numpy(rewards)
                y = rewards + algo.gamma * next_values * mask
                total_errors += ((values - y) ** 2).tolist()

        return float(np.mean(total_errors))


def d3_dataset_from_trajs(trajs, cfg):
    import d3rlpy as d3
    obs_keys = cfg.obs_keys
    all_obs, all_actions, all_rewards, all_terminals = [], [], [], []
    for traj in trajs:
        # if traj["state"][0]["option_time"] != traj["state"][1]["option_time"]:
            # XXX some bug
            # continue
        if cfg.ignore_intra_option_states:
            # duplicate state at each step, but for the first skill both
            # duplicates have option_time = 0
            option_reward = None
            for state, action, reward in zip(
                traj["state"][1:], traj["action"][1:], traj["reward"][1:]
            ):
                if not cfg.privileged_critic:
                    state = state['obs']
                if state["option_time"][0] == 0:
                    obs = np.concatenate(
                        [
                            np.array(state[key]).flatten()
                            for key in obs_keys
                            # if key in state
                        ]
                    )
                    all_obs.append(obs)
                    all_actions.append(action)
                    if option_reward is not None:
                        all_rewards.append(option_reward)
                    option_reward = reward
                    all_terminals.append(False)
                else:
                    if option_reward is None or reward is None:
                        __import__("ipdb").set_trace()
                    option_reward += reward
            all_rewards.append(option_reward)
            all_terminals[-1] = True
        else:
            if cfg.privileged_critic:
                obs = [
                    np.concatenate(
                        [
                            np.array(state[key]).flatten()
                            for key in obs_keys
                            # if key in state
                        ]
                    )
                    for state in traj["state"]
                ]
            else:
                obs = [
                    np.concatenate(
                        [
                            np.array(state['obs'][key]).flatten()
                            for key in obs_keys
                            # if key in state['obs']
                        ]
                    )
                    for state in traj["state"]
                ]
            # d3rlpy convention is now same as normal (s0, a0, r0, done0, s1,...)
            # see https://github.com/takuseno/d3rlpy/issues/166
            # for earlier convention see https://github.com/takuseno/d3rlpy/issues/98
            if cfg.drop_last_obs:
                obs = np.array(obs[:-1])
                actions = traj["action"]
                rewards = traj["reward"]
            else:
                obs = np.array(obs)
                actions = traj["action"] + [0]  # dummy actions
                rewards = traj["reward"] + [0]
            L = len(obs)
            all_obs.append(obs)
            all_actions.append(np.array(actions))  # dummy action
            all_rewards.append(np.array(rewards))  # reward
            terminals = np.zeros(L)
            terminals[-1] = True
            all_terminals.append(terminals)

    if cfg.ignore_intra_option_states:
        all_obs = np.array(all_obs)
        all_actions = np.array(all_actions)
        all_rewards = np.array(all_rewards)
        all_terminals = np.array(all_terminals)
    else:
        all_obs = np.concatenate(all_obs)
        all_actions = np.concatenate(all_actions)
        all_rewards = np.concatenate(all_rewards)
        all_terminals = np.concatenate(all_terminals)

    logger.info(f"  #transitions: {len(all_obs)}")

    dataset = d3.dataset.MDPDataset(
        observations=all_obs,
        actions=all_actions,
        rewards=all_rewards,
        terminals=all_terminals,
    )
    return dataset
