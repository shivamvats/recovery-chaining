"""
Adapted from stable_baselines3/ppo/ppo.py
"""

from heapq import heappush, heappop, heapify
import logging
import warnings
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union
from os.path import join
import random
import sys
import time

import d3rlpy as d3
import numpy as np
import torch as th
from gymnasium import spaces
import matplotlib.colors as mcolors
from torch.nn import functional as F
from omegaconf import OmegaConf
import wandb

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import obs_as_tensor, safe_mean

# from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from .on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import (
    ActorCriticCnnPolicy,
    ActorCriticPolicy,
    BasePolicy,
    MultiInputActorCriticPolicy,
)
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.vec_env import sync_envs_normalization

from belief_srs.utils.d3_utils import (
    ImprovedTDErrorEvaluator,
    ReturnErrorEvaluator,
    d3_dataset_from_trajs,
)
from belief_srs.utils import pkl_load, to_absolute_path
from belief_srs.envs.utils import reset_venv, venv_method

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

SelfPPO = TypeVar("SelfPPO", bound="PPOTermQ")


class FailureBuffer:
    def __init__(self):
        self.data = []
        self.priority = []

    def sample(self, n):
        sample_probs = np.array(self.priority) / np.sum(self.priority)
        sample_idxs = np.random.choice(len(self.data), n, p=sample_probs)
        return [self.data[idx] for idx in sample_idxs]

    def add(self, state, priority=1):
        self.data.append(state)
        self.priority.append(priority)


class PPOTermQ(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        potential_fn=None,
        num_nominal_actions=1,
        cfg=None,
        rollout_env=None,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        self.potential_fn = potential_fn
        self.num_nominal_actions = num_nominal_actions
        self.cfg = cfg
        self.rl_cfg = cfg.rl["PPOTermQ"]
        # critic
        self.critic_update_freq = self.rl_cfg.critic.critic_update_freq
        self.n_nominal_rollouts = np.ceil(
            self.rl_cfg.critic.n_nominal_rollouts / self.n_envs
        ).astype(int)
        self.rollout_env = rollout_env

        self.nominal_action_offset = self.env.get_attr(
            "num_primitive_actions", indices=0
        )[0]

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # buffer to store states where nominal actions are triggered
        self.nominal_state_actions = []
        # heapify(self.nominal_state_actions)

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, (
                    "`clip_range_vf` must be positive, "
                    "pass `None` to deactivate vf clipping"
                )

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        assert self._last_true_state is not None, "No previous true state was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        # env.env_method("set_execute_nominal_actions", True)

        all_actions, all_values, all_true_states = [], [], []
        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()
            all_actions.extend(list(actions.flatten()))
            all_values.extend(list(values.cpu().numpy().flatten()))

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(
                        actions, self.action_space.low, self.action_space.high
                    )

            true_states = env.env_method(
                "observe_true_state", mj_state=True, indices=list(range(env.num_envs))
            )
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            new_true_states = [info["true_state"] for info in infos]

            all_true_states.extend(list(new_true_states))
            if "timesteps" in infos[0]:
                self.num_timesteps += sum(info["timesteps"] for info in infos)
            else:
                self.num_timesteps += env.num_envs

            for i, info in enumerate(infos):
                if info["nominal_action"]:
                    # XXX if priority of two elements is same then heapq will
                    # compare the elements which will trigger an error
                    # heappush(
                    # fixed size queue
                    self.nominal_state_actions.insert(
                        0,
                        (
                            -len(self.nominal_state_actions),
                            true_states[i],
                            clipped_actions[i],
                        ),
                    )
                    # discard older states
                    buffer_size = self.rl_cfg["critic"].start_buffer_size
                    if buffer_size > 0:
                        self.nominal_state_actions = self.nominal_state_actions[
                            :buffer_size
                        ]
                    # logger.info(
                    # f"  Nominal action triggered. Nominal buffer size: {len(self.nominal_state_actions)}"
                    # )

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                true_state=self._last_true_state,
            )
            # print("idx, term action, actions, value: ",
            # n_steps, env.env_method("get_term_action_ids", indices=0), actions, values)
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_true_state = new_true_states
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        self._last_values = values
        self._last_dones = dones
        # XXX called after updating rewards with critic
        # rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def collect_nominal_rollouts(
        self,
        env: VecEnv,
        n_rollouts: int,
        callback=None,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollouts: Number of trajectories to collect
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        # Switch to eval mode (this affects batch norm / dropout)
        logger.debug(f"  Collecting {n_rollouts} nominal rollouts")
        self.policy.set_training_mode(False)

        sync_envs_normalization(self.env, env)

        # rollout_buffer.reset()
        # Sample new weights for the state dependent exploration

        # callback.on_rollout_start()
        # env.env_method("set_execute_nominal_actions", True)

        self._new_nominal_transitions = 0
        Q_predicted, Q_mc = [], []
        for i in range(n_rollouts):
            logger.info(f"  nominal start states: {len(self.nominal_state_actions)}")
            # if len(self.nominal_state_actions) < N:
            if len(self.nominal_state_actions) == 0:
                logger.info("  -------------")
                logger.info("  No more nominal start states!")
                logger.info("  -------------")
                break

            # start_state_actions = [
            # heappop(self.nominal_state_actions)
            # for _ in range(min(env.num_envs, len(self.nominal_state_actions)))
            # ]
            # uniformly random sampling
            N = min(env.num_envs, len(self.nominal_state_actions))
            start_state_actions = []
            for _ in range(N):
                start_state_actions.append(
                    self.nominal_state_actions.pop(
                        random.randrange(len(self.nominal_state_actions))
                    )
                )
            env_start_states = [state for (_, state, _) in start_state_actions]
            env_state = [state for (_, state, _) in start_state_actions]
            env_actions = [action for (_, _, action) in start_state_actions]

            # reset to state where nominal needs to be evaluated
            all_obs = reset_venv(env, env_start_states, indices=list(range(N)))
            list_of_args = [(action, obs) for action, obs in zip(env_actions, all_obs)]
            # obsx, rews, dones, infos = venv_method(env, "step", list_of_args)
            logger.info("  Nominal rollout: stepping")
            results = venv_method(env, "step", list_of_args, indices=list(range(N)))

            logger.info("  Nominal rollout: adding to buffer")
            rews = []
            # __import__('ipdb').set_trace()
            for env_id, result in enumerate(results):
                _, _, _, _, info = result
                # __import__('ipdb').set_trace()
                logger.debug(f"  nominal chain reward: {info['hist']['reward']}")
                logger.debug(f"  nominal chain skills: {info['hist']['skill']}")
                for state, rew, skill_id in zip(
                    info["hist"]["state"], info["hist"]["reward"], info["hist"]["skill"]
                ):
                    rews.append(rew)
                    self._new_nominal_transitions += 1
                    if self.rl_cfg["critic"].ignore_intra_option_states:
                        raise NotImplementedError
                    else:
                        self.critic_buffer.append(
                            self.flatten_critic_state(state),
                            # self._offset_nominal_actions(env_actions[env_id]),
                            skill_id,
                            float(rew),
                        )
                    # TODO
                    # update reward in self.rollout_buffer

                if "timesteps" in info:
                    self.num_timesteps += info["timesteps"]
                else:
                    self.num_timesteps += env.num_envs

                # nominal action is terminal action
                terminated = True
                self.critic_buffer.clip_episode(terminated)

            logger.debug(f"  Critic buffer size: {self.critic_buffer.transition_count}")
            Q_mc.extend(rews)
            # predicted Q
            X_batch = np.array(
                [self.flatten_critic_state(state) for state in env_state]
            )
            A_batch = self._offset_nominal_actions(np.array(env_actions))
            # predict terminal Q values
            # Q_batch_predicted = self.critic.predict_value(X_batch, A_batch)
            Q_batch_predicted_ensemble = self.critic.predict_value_ensemble(
                X_batch, A_batch
            )
            Q_batch_predicted = np.mean(Q_batch_predicted_ensemble, axis=0)
            Q_predicted.extend(list(Q_batch_predicted))

        if n_rollouts:
            self.logger.record("rollout/nominal_rew_mean", np.mean(Q_mc))
            self.logger.record(
                "rollout/critic_overestimation", np.mean(Q_predicted) - np.mean(Q_mc)
            )

        # __import__("ipdb").set_trace()
        # true_states = env.env_method(
        # "observe_true_state", mj_state=False, indices=list(range(env.num_envs))
        # )
        # new_obs, rewards, dones, infos = env.step(clipped_actions)
        # new_true_states = [info["true_state"] for info in infos]

        # all_true_states.extend(list(new_true_states))

        # self.num_timesteps += len(results)

        # Give access to local variables
        # callback.update_locals(locals())
        # if not callback.on_step():
        # return False

        # self._update_info_buffer(infos)
        # n_trajs += 1

        # print("idx, term action, actions, value: ",
        # n_steps, env.env_method("get_term_action_ids", indices=0), actions, values)
        # self._last_obs = new_obs  # type: ignore[assignment]
        # self._last_true_state = new_true_states
        # self._last_episode_starts = dones

        # callback.update_locals(locals())

        # callback.on_rollout_end()

        # env.env_method("set_execute_nominal_actions", False)

        return True

    def initialize_nominal_critic(self, nom_trajs):
        critic_cfg = self.rl_cfg["critic"]
        mdp_dataset = d3_dataset_from_trajs(nom_trajs, critic_cfg)
        if critic_cfg.init:
            if critic_cfg.monte_carlo:
                raise NotImplementedError
            else:
                self.critic_buffer = d3.dataset.ReplayBuffer(
                    d3.dataset.buffers.FIFOBuffer(limit=critic_cfg.replay_buffer_size),
                    episodes=mdp_dataset.episodes,
                )
        else:
            if critic_cfg.monte_carlo:
                raise NotImplementedError
            else:
                self.critic_buffer = d3.dataset.ReplayBuffer(
                    d3.dataset.buffers.FIFOBuffer(limit=critic_cfg.replay_buffer_size),
                    episodes=mdp_dataset.episodes[:2],
                )

        if critic_cfg.monte_carlo:
            raise NotImplementedError

        else:
            if critic_cfg.algo == "q_eval":
                from .offpolicy_q_evaluation import QEvaluatorConfig

                self.critic = QEvaluatorConfig(
                    learning_rate=critic_cfg.lr,
                    batch_size=critic_cfg.batch_size,
                    n_critics=critic_cfg.n_critics,
                    observation_scaler=d3.preprocessing.StandardObservationScaler(),
                ).create(device="cuda:0")
            elif critic_cfg.algo == "ddqn":
                self.critic = d3.algos.DoubleDQNConfig(
                    learning_rate=critic_cfg.lr,
                    batch_size=critic_cfg.batch_size,
                    target_update_interval=critic_cfg.target_update_interval,
                    n_critics=critic_cfg.n_critics,
                    # encoder_factory=VectorEncoderFactory(hidden_units=cfg.ddqn_hidden_units),
                    observation_scaler=d3.preprocessing.StandardObservationScaler(),
                ).create(device="cuda:0")

            elif critic_cfg.algo == "cql":
                self.critic = d3.algos.DiscreteCQLConfig(
                    alpha=critic_cfg.cql_alpha,
                    learning_rate=critic_cfg.lr,
                    batch_size=critic_cfg.batch_size,
                    target_update_interval=critic_cfg.target_update_interval,
                    n_critics=critic_cfg.n_critics,
                    observation_scaler=d3.preprocessing.StandardObservationScaler(),
                ).create(device="cuda:0")

            else:
                raise NotImplementedError
        self._new_nominal_transitions = self.critic_buffer.transition_count
        return self.critic

    def flatten_critic_state(self, state):
        if self.rl_cfg["critic"].privileged_critic:
            obs_arr = np.concatenate(
                [np.array(state[key]).flatten() for key in self.rl_cfg.critic.obs_keys]
            )
        else:
            obs_arr = np.concatenate(
                [
                    np.array(state["obs"][key]).flatten()
                    for key in self.rl_cfg.critic.obs_keys
                ]
            )
        return obs_arr

    def update_critic(self, n_steps, callback=None):
        critic_cfg = self.rl_cfg["critic"]
        if n_steps < 0:
            n_steps = np.ceil(np.abs(n_steps) * self._new_nominal_transitions).astype(
                int
            )
            n_steps_per_epoch = n_steps
            logger.info(f"  Updating critic with {n_steps} gradient steps")
        else:
            n_steps_per_epoch = critic_cfg.n_steps_per_epoch

        for epoch, metrics in self.critic.fitter(
            self.critic_buffer,
            n_steps=n_steps,
            n_steps_per_epoch=n_steps_per_epoch,
            evaluators={
                "td_error": ImprovedTDErrorEvaluator(),
                "monte_carlo_error": ReturnErrorEvaluator(),
                "value_mean": d3.metrics.AverageValueEstimationEvaluator(),
                "init_value": d3.metrics.InitialStateValueEstimationEvaluator(),
            },
            logger_adapter=d3.logging.TensorboardAdapterFactory(root_dir="."),
            with_timestamp=False,
        ):
            # wandb.run.log(
            # {
            # f"q_value_train": plot_error(critic),
            # f"q_value_test": plot_error(critic),
            # }
            # )
            self.critic.save("critic_init.d3")
            wandb.save("critic_init.d3")

        return self.critic

    def update_buffer_with_critic(self):
        nom_action_ids = self.env.get_attr("term_actions", indices=0)[0]
        nom_is, nom_js = [], []
        for b in range(self.rollout_buffer.buffer_size):
            for env_id in range(self.rollout_buffer.n_envs):
                if self.rollout_buffer.actions[b, env_id].item() in nom_action_ids:
                    nom_is.append(b)
                    nom_js.append(env_id)

        if len(nom_is) <= 1:
            return

        # X_batch = self.rollout_buffer.observations[nom_is, nom_js]
        X_batch = np.array(
            [
                self.flatten_critic_state(self.rollout_buffer.true_states[i][j])
                for i, j in zip(nom_is, nom_js)
            ]
        )

        A_batch = self.rollout_buffer.actions[nom_is, nom_js]
        # offset
        A_batch = self._offset_nominal_actions(A_batch)
        # predict terminal Q values
        Q_batch = self.critic.predict_value(X_batch, A_batch)
        if self.rl_cfg["critic"].clip_critic:
            Q_batch = np.clip(
                Q_batch,
                self.rl_cfg["critic"].critic_clip_range[0],
                self.rl_cfg["critic"].critic_clip_range[1],
            )
        self.logger.record("rollout/term_q_mean", np.mean(Q_batch))
        for i, (b, env_id) in enumerate(zip(nom_is, nom_js)):
            self.rollout_buffer.rewards[b, env_id] = Q_batch[i]

        # TODO
        # return returns and advantages
        self.rollout_buffer.compute_returns_and_advantage(
            self._last_values, self._last_dones
        )

    def _offset_nominal_actions(self, actions):
        return actions - self.nominal_action_offset

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ):
        total_timesteps, callback = super()._setup_learn(
            total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar
        )

        self._last_true_state = list(
            self.env.env_method(
                "observe_true_state",
                mj_state=False,
                indices=list(range(self.env.num_envs)),
            )
        )
        return total_timesteps, callback

    def learn(
        self: SelfPPO,
        total_timesteps: int,
        cfg=None,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPO:
        iteration = 0

        wandb.define_metric("timesteps")
        wandb.define_metric("rollout/*", step_metric="timesteps")
        wandb.define_metric("eval/*", step_metric="timesteps")

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        # load nominal trajectories
        nom_trajs = pkl_load(
            join(to_absolute_path(cfg.data_dir), cfg.failures.demo_filename)
        )
        if "pos" in nom_trajs:
            nom_trajs = nom_trajs["pos"] + nom_trajs["neg"]
        critic_cfg = self.rl_cfg["critic"]

        # load nominal dataset
        self.initialize_nominal_critic(nom_trajs)
        if critic_cfg.init:
            self.update_critic(critic_cfg.init_n_steps)

        while self.num_timesteps < total_timesteps:
            # recovery policy rollouts without nom rollout
            continue_training = self.collect_rollouts(
                self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps
            )

            if (
                self.rl_cfg.update_critic
                and self.num_timesteps % self.critic_update_freq == 0
            ):
                # collect additional nominal rollouts and update critic
                # reset to state isn D_nom and rollout nominal

                if self.rl_cfg.critic.nominal_rollout_ratio >= 1:
                    n_nominal_rollouts = np.ceil(
                        len(self.nominal_state_actions)
                        / self.n_envs
                        / self.rl_cfg.critic.nominal_rollout_ratio
                    ).astype(int)
                else:
                    n_nominal_rollouts = self.n_nominal_rollouts

                self.collect_nominal_rollouts(
                    self.rollout_env,
                    # callback,
                    n_rollouts=n_nominal_rollouts,
                )
                # reset buffer of evaluated edges
                if critic_cfg.reset_nominal_buffer_online:
                    self.nominal_state_actions = []
                # not needed after separate rollout env
                # self._last_obs = self.env.reset()  # type: ignore[assignment]
                self.update_critic(critic_cfg.n_steps, callback)

            # use critic to predictr nominal action rewards
            self.update_buffer_with_critic()

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                time_elapsed = max(
                    (time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon
                )
                fps = int(
                    (self.num_timesteps - self._num_timesteps_at_start) / time_elapsed
                )
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    wandb.log(
                        {
                            "rollout/ep_rew_mean": safe_mean(
                                [ep_info["r"] for ep_info in self.ep_info_buffer]
                            ),
                            "rollout/ep_len_mean": safe_mean(
                                [ep_info["l"] for ep_info in self.ep_info_buffer]
                            ),
                            "timesteps": self.num_timesteps,
                        }
                    )
                self.logger.record("time/fps", fps)
                self.logger.record(
                    "time/time_elapsed", int(time_elapsed), exclude="tensorboard"
                )
                self.logger.record(
                    "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
                )
                self.logger.dump(step=self.num_timesteps)

            if self.num_timesteps >= self.rl_cfg.actor_freeze_steps:
                self.train()

        callback.on_training_end()

        return self
