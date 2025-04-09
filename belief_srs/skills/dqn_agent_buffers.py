"""
Note: Copied from stable-baselines3-contrib
"""

from functools import partial
from typing import (
    Callable,
    Generator,
    Optional,
    Tuple,
    Union,
    NamedTuple,
    List,
    Dict,
    Any,
)

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import BaseBuffer, ReplayBuffer


class RNNStates(NamedTuple):
    vf: Tuple[torch.Tensor, ...]


class RecurrentReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    lstm_hidden_states: torch.Tensor
    lstm_cell_states: torch.Tensor
    episode_starts: torch.Tensor
    # mask: torch.Tensor


class RecurrentReplayBuffer(ReplayBuffer):
    """
    Rollout buffer that also stores the LSTM cell and hidden states.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param hidden_state_shape: Shape of the buffer that will collect lstm states
        (n_steps, lstm.num_layers, n_envs, lstm.hidden_size)
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        hidden_state_shape: Tuple[int, int, int, int],
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
    ):
        self.hidden_state_shape = hidden_state_shape
        self.seq_start_indices, self.seq_end_indices = None, None
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
        )

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        self.hidden_states_vf = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self.cell_states_vf = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(self, *args, lstm_hidden_states, lstm_cell_states, episode_start: np.ndarray, **kwargs) -> None:
        """
        :param hidden_states: LSTM cell and hidden state
        """
        self.hidden_states_vf[self.pos] = np.array(lstm_hidden_states[0].cpu().numpy())
        self.cell_states_vf[self.pos] = np.array(lstm_cell_states[1].cpu().numpy())
        self.episode_starts[self.pos] = np.array(episode_start)

        super().add(*args, **kwargs)
        __import__('ipdb').set_trace()

    # def get(
        # self, batch_size: Optional[int] = None
    # ) -> Generator[RecurrentReplayBufferSamples, None, None]:
        # assert self.full, "Replay buffer must be full before sampling from it"
    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):

        # Prepare the data
        if not self.generator_ready:
            # hidden_state_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)
            # swap first to (self.n_steps, self.n_envs, lstm.num_layers, lstm.hidden_size)
            for tensor in [
                "hidden_states_vf",
                "cell_states_vf",
            ]:
                self.__dict__[tensor] = self.__dict__[tensor].swapaxes(1, 2)

            # flatten but keep the sequence order
            # 1. (n_steps, n_envs, *tensor_shape) -> (n_envs, n_steps, *tensor_shape)
            # 2. (n_envs, n_steps, *tensor_shape) -> (n_envs * n_steps, *tensor_shape)
            for tensor in [
                "observations",
                "actions",
                "values",
                # "returns",
                "hidden_states_vf",
                "cell_states_vf",
                "episode_starts",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        # Sampling strategy that allows any mini batch size but requires
        # more complexity and use of padding
        # Trick to shuffle a bit: keep the sequence order
        # but split the indices in two
        split_index = np.random.randint(self.buffer_size * self.n_envs)
        indices = np.arange(self.buffer_size * self.n_envs)
        indices = np.concatenate((indices[split_index:], indices[:split_index]))

        env_change = np.zeros(self.buffer_size * self.n_envs).reshape(
            self.buffer_size, self.n_envs
        )
        # Flag first timestep as change of environment
        env_change[0, :] = 1.0
        env_change = self.swap_and_flatten(env_change)

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            batch_inds = indices[start_idx : start_idx + batch_size]
            yield self._get_samples(batch_inds, env_change)
            start_idx += batch_size

    # def _get_samples(
        # self,
        # batch_inds: np.ndarray,
        # env_change: np.ndarray,
        # env: Optional[VecNormalize] = None,
    # ) -> RecurrentReplayBufferSamples:
        # # Retrieve sequence starts and utility function
        # self.seq_start_indices, self.pad, self.pad_and_flatten = create_sequencers(
            # self.episode_starts[batch_inds], env_change[batch_inds], self.device
        # )

        # # Number of sequences
        # n_seq = len(self.seq_start_indices)
        # max_length = self.pad(self.actions[batch_inds]).shape[1]
        # padded_batch_size = n_seq * max_length
        # # We retrieve the lstm hidden states that will allow
        # # to properly initialize the LSTM at the beginning of each sequence
        # lstm_states_pi = (
            # # 1. (n_envs * n_steps, n_layers, dim) -> (batch_size, n_layers, dim)
            # # 2. (batch_size, n_layers, dim)  -> (n_seq, n_layers, dim)
            # # 3. (n_seq, n_layers, dim) -> (n_layers, n_seq, dim)
            # self.hidden_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
            # self.cell_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        # )
        # lstm_states_vf = (
            # # (n_envs * n_steps, n_layers, dim) -> (n_layers, n_seq, dim)
            # self.hidden_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
            # self.cell_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        # )
        # lstm_states_vf = (
            # self.to_torch(lstm_states_vf[0]).contiguous(),
            # self.to_torch(lstm_states_vf[1]).contiguous(),
        # )

        # return RecurrentReplayBufferSamples(
            # # (batch_size, obs_dim) -> (n_seq, max_length, obs_dim) -> (n_seq * max_length, obs_dim)
            # observations=self.pad(self.observations[batch_inds]).reshape(
                # (padded_batch_size, *self.obs_shape)
            # ),
            # actions=self.pad(self.actions[batch_inds]).reshape(
                # (padded_batch_size,) + self.actions.shape[1:]
            # ),
            # old_values=self.pad_and_flatten(self.values[batch_inds]),
            # old_log_prob=self.pad_and_flatten(self.log_probs[batch_inds]),
            # advantages=self.pad_and_flatten(self.advantages[batch_inds]),
            # returns=self.pad_and_flatten(self.returns[batch_inds]),
            # lstm_states=RNNStates(lstm_states_pi, lstm_states_vf),
            # episode_starts=self.pad_and_flatten(self.episode_starts[batch_inds]),
            # mask=self.pad_and_flatten(np.ones_like(self.returns[batch_inds])),
        # )


def pad(
    seq_start_indices: np.ndarray,
    seq_end_indices: np.ndarray,
    device: torch.device,
    tensor: np.ndarray,
    padding_value: float = 0.0,
) -> torch.Tensor:
    """
    Chunk sequences and pad them to have constant dimensions.

    :param seq_start_indices: Indices of the transitions that start a sequence
    :param seq_end_indices: Indices of the transitions that end a sequence
    :param device: PyTorch device
    :param tensor: Tensor of shape (batch_size, *tensor_shape)
    :param padding_value: Value used to pad sequence to the same length
        (zero padding by default)
    :return: (n_seq, max_length, *tensor_shape)
    """
    # Create sequences given start and end
    seq = [
        th.tensor(tensor[start : end + 1], device=device)
        for start, end in zip(seq_start_indices, seq_end_indices)
    ]
    return th.nn.utils.rnn.pad_sequence(
        seq, batch_first=True, padding_value=padding_value
    )


def pad_and_flatten(
    seq_start_indices: np.ndarray,
    seq_end_indices: np.ndarray,
    device: torch.device,
    tensor: np.ndarray,
    padding_value: float = 0.0,
) -> torch.Tensor:
    """
    Pad and flatten the sequences of scalar values,
    while keeping the sequence order.
    From (batch_size, 1) to (n_seq, max_length, 1) -> (n_seq * max_length,)

    :param seq_start_indices: Indices of the transitions that start a sequence
    :param seq_end_indices: Indices of the transitions that end a sequence
    :param device: PyTorch device (cpu, gpu, ...)
    :param tensor: Tensor of shape (max_length, n_seq, 1)
    :param padding_value: Value used to pad sequence to the same length
        (zero padding by default)
    :return: (n_seq * max_length,) aka (padded_batch_size,)
    """
    return pad(
        seq_start_indices, seq_end_indices, device, tensor, padding_value
    ).flatten()


def create_sequencers(
    episode_starts: np.ndarray,
    env_change: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, Callable, Callable]:
    """
    Create the utility function to chunk data into
    sequences and pad them to create fixed size tensors.

    :param episode_starts: Indices where an episode starts
    :param env_change: Indices where the data collected
        come from a different env (when using multiple env for data collection)
    :param device: PyTorch device
    :return: Indices of the transitions that start a sequence,
        pad and pad_and_flatten utilities tailored for this batch
        (sequence starts and ends indices are fixed)
    """
    # Create sequence if env changes too
    seq_start = np.logical_or(episode_starts, env_change).flatten()
    # First index is always the beginning of a sequence
    seq_start[0] = True
    # Retrieve indices of sequence starts
    seq_start_indices = np.where(seq_start == True)[0]  # noqa: E712
    # End of sequence are just before sequence starts
    # Last index is also always end of a sequence
    seq_end_indices = np.concatenate(
        [(seq_start_indices - 1)[1:], np.array([len(episode_starts)])]
    )

    # Create padding method for this minibatch
    # to avoid repeating arguments (seq_start_indices, seq_end_indices)
    local_pad = partial(pad, seq_start_indices, seq_end_indices, device)
    local_pad_and_flatten = partial(
        pad_and_flatten, seq_start_indices, seq_end_indices, device
    )
    return seq_start_indices, local_pad, local_pad_and_flatten


