from collections import namedtuple
from copy import deepcopy
import logging
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from stable_baselines3.common.policies import ContinuousCritic
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from belief_srs.utils import *
import wandb

logger = logging.getLogger(__name__)


class DataBuffer:
    def __init__(
        self,
        obs_keys,
        num_actions,
        batch_size=50,
        discount=0.99,
        shuffle=True,
        device="cpu",
    ):
        self.obs_keys = obs_keys
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.discount = discount
        self.shuffle = shuffle
        self.device = torch.device(device)

        self.data = []
        self.priority = []
        self.episodes = []
        self.obs_scaler = None

    def obs_dim(self):
        return self.data[0]["state"].shape[0]

    def add_episode(self, ep):
        try:
            obsx = [
                np.concatenate([np.array(transition['obs'][key]).flatten() for key in self.obs_keys])
                for transition in ep["state"]
            ]
        except KeyError:
            obsx = [
                np.concatenate([np.array(transition[key]).flatten() for key in self.obs_keys])
                for transition in ep["state"]
            ]
        rews = np.array(ep["reward"])
        actions = np.array(ep["action"])
        returns = rews_to_returns(rews, discount=self.discount)

        # no action or reward for terminal state
        # done = True for terminal state
        assert len(obsx) == len(rews) + 1

        episode = dict(state=obsx, action=actions, reward=rews, returns=returns)
        self.episodes.append(episode)

        # collect (s_t, a_t, r_t, done_t, s_t+1, return_t) tuples
        # ignore terminal state ((not required for bootstrapping)
        # see discussion at https://github.com/takuseno/d3rlpy/issues/98
        for i in range(0, len(rews)):
            if i == len(rews) - 1:
                done = True
            else:
                done = False
            transition = dict(
                state=deepcopy(obsx[i]),
                action=actions[i],
                reward=rews[i],
                next_state=deepcopy(obsx[i + 1]),
                done=done,
                returns=returns[i],
            )
            # transition = dict(
                # state=deepcopy(obsx[i]),
                # action=actions[i]
                # if not done
                # else np.random.randint(0, self.num_actions),  # dummy action
                # reward=rews[i] if not done else 0,
                # next_state=deepcopy(obsx[i + 1])
                # if not done
                # else deepcopy(obsx[i]),  # self-loop
                # done=done,
                # returns=returns[i] if not done else 0,
            # )
            self.add_transition(transition)
        # rews = [tran['reward'] for tran in self.data]
        # states = [tran['state'] for tran in self.data]
        # rets = [tran['returns'] for tran in self.data]
        # dones = [tran['done'] for tran in self.data]

    def add_transition(self, transition):
        transition["index"] = len(self.data)
        self.data.append(transition)
        self.priority.append(1)

    def fit_scaler(self):
        obs_scaler = StandardScaler()
        X = np.array([transition["state"] for transition in self.data])
        obs_scaler.fit(X)
        self.obs_scaler = obs_scaler
        logger.info(f"  Observation shape: {X.shape}")

    def sample(self):
        batch = random.choices(self.data, weights=self.priority, k=self.batch_size)
        batch = self._batch_to_tensor(batch)
        return batch

    def __iter__(self):
        self.iter = 0
        if self.shuffle:
            random.shuffle(self.data)
        return self

    def __next__(self):
        if self.iter < len(self.data) // self.batch_size:
            batch = self.data[
                self.iter * self.batch_size : (self.iter + 1) * self.batch_size
            ]
            batch = self._batch_to_tensor(batch)

            self.iter += 1
            return batch
        else:
            raise StopIteration

    def _batch_to_tensor(self, batch):
        states = torch.tensor(
            self.obs_scaler.transform(
                np.array([transition["state"] for transition in batch])
            ),
            dtype=torch.float32,
        ).to(self.device)
        actions = torch.tensor(
            [transition["action"] for transition in batch], dtype=torch.int64
        ).to(self.device)
        next_states = torch.tensor(
            self.obs_scaler.transform(
                np.array([transition["next_state"] for transition in batch])
            ),
            dtype=torch.float32,
        ).to(self.device)
        rewards = torch.tensor(
            [transition["reward"] for transition in batch], dtype=torch.float32
        ).to(self.device)
        dones = torch.tensor(
            [transition["done"] for transition in batch], dtype=torch.int
        ).to(self.device)
        returns = torch.tensor(
            [transition["returns"] for transition in batch], dtype=torch.float32
        ).to(self.device)
        index = torch.tensor([transition["index"] for transition in batch], dtype=torch.int)

        batch = dict(
            state=states,
            action=actions,
            next_state=next_states,
            reward=rewards,
            done=dones,
            returns=returns,
            index=index
        )
        return batch


class NominalValueFunction:
    def __init__(self, cfg, device="cpu"):
        self.q_value = cfg.q_value
        self.obs_keys = cfg.obs_keys
        self.monte_carlo = cfg.monte_carlo
        self.td_steps = cfg.td_steps
        self.num_epochs = cfg.num_epochs
        self.lr = cfg.lr
        self.batch_size = cfg.batch_size
        self.discount = cfg.discount
        self.device = torch.device(device)

    def __call__(self, states):
        device = self.device
        X = []
        # extract relevant features
        try:
            # observations
            for state in states:
                X.append(np.concatenate([np.array(state['obs'][key]).flatten() for key in self.obs_keys]))
        except KeyError:
            # privileged info
            for state in states:
                X.append(np.concatenate([np.array(state[key]).flatten() for key in self.obs_keys]))
        X = np.array(X)
        # standardize
        X = self.scaler.transform(X)

        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = self.model(X).cpu().numpy()

        if not self.q_value:
            preds = preds.flatten()

        return preds

    def predict_value(self, states, actions=None):
        q_values = self.__call__(states)
        if self.q_value:
            # choose action indices
            return q_values[np.arange(q_values.shape[0]), actions]
        else:
            return q_values

    def train(self, trajs):
        logger.info("Training value function")
        logger.info("  Config:")
        logger.info(f"    Monte Carlo: {self.monte_carlo}")
        logger.info(f"    Discount: {self.discount}")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        def create_replay_buffer(trajs):
            # train test split
            train_trajs, test_trajs = train_test_split(
                trajs, test_size=0.1, shuffle=True, random_state=42
            )

            nactions = len(
                np.unique(np.concatenate([traj["action"] for traj in train_trajs]))
            )
            self.nactions = nactions

            train_buffer = DataBuffer(
                self.obs_keys,
                num_actions=nactions,
                batch_size=self.batch_size,
                discount=self.discount,
                shuffle=True,
                device="cuda:0",
            )
            [train_buffer.add_episode(ep) for ep in train_trajs]
            train_buffer.fit_scaler()

            test_buffer = DataBuffer(
                self.obs_keys,
                num_actions=nactions,
                batch_size=self.batch_size,
                discount=self.discount,
                shuffle=True,
                device="cuda:0",
            )
            [test_buffer.add_episode(ep) for ep in test_trajs]
            test_buffer.obs_scaler = train_buffer.obs_scaler

            logger.info("  Train buffer size: {}".format(len(train_buffer.data)))
            logger.info("  Test buffer size: {}".format(len(test_buffer.data)))
            return train_buffer, test_buffer, train_buffer.obs_scaler

        train_buffer, test_buffer, scaler = create_replay_buffer(trajs)
        self.scaler = scaler
        in_dim = train_buffer.obs_dim()
        if self.q_value:
            out_dim = train_buffer.num_actions
        else:
            out_dim = 1

        # Define the model
        # same as stable baselines3
        Q_value = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        ).to(device)

        # TODO ortho_init

        # loss_fn = nn.MSELoss()  # mean square error
        loss_fn = nn.SmoothL1Loss()  # Huber loss

        def evaluate_model(T_batch):
            S_batch = T_batch["state"]
            A_batch = T_batch["action"]
            S_prime_batch = T_batch["next_state"]
            R_batch = T_batch["reward"]
            done_batch = T_batch["done"]
            G_batch = T_batch["returns"]

            # for fitted q iteration
            Q_s_batch = Q_value(S_batch)
            if self.q_value:
                Q_batch = torch.gather(
                    Q_s_batch, dim=1, index=A_batch.unsqueeze(1)
                ).squeeze()
            else:
                Q_batch = Q_s_batch.squeeze()

            # for fitted value iteration
            # V_batch = torch.argmax(Q_batch, dim=1)

            if self.monte_carlo:
                # loss = G(s) - v(s)
                loss = loss_fn(G_batch, Q_batch)
            else:
                if self.q_value:
                    V_prime_batch = torch.argmax(Q_value(S_prime_batch), dim=1)
                else:
                    V_prime_batch = Q_value(S_prime_batch).squeeze()
                # loss = r + v(s') - v(s)
                loss = loss_fn(
                    R_batch + (1 - done_batch) * self.discount * V_prime_batch, Q_batch
                )
            return loss

        # optimizer = optim.Adam(Q_value.parameters(), self.lr, eps=1e-5)
        optimizer = optim.Adam(Q_value.parameters(), self.lr)

        # Hold the best model
        best_value_loss = np.inf  # init to infinity
        best_weights = None
        history = []

        logger.info("  Training value function")
        # training loop
        no_improvement = 0
        for epoch in range(self.num_epochs):
            Q_value.train()
            # np.random.shuffle(T_train)
            # with tqdm(batch_start, unit="batch", mininterval=0, disable=False) as bar:
            train_loss = []
            for batch in train_buffer:
                # bar.set_description(f"Epoch {epoch}")
                # for start in bar:
                # take a batch
                # value update
                # T_batch = T_train[start : start + batch_size]  # .to(device)
                loss = evaluate_model(batch)

                # backward pass
                optimizer.zero_grad()
                loss.backward()

                # update weights
                optimizer.step()
                # print progress
                # bar.set_postfix(mse=float(loss))
                train_loss.append(loss.item())

            train_loss = np.mean(train_loss)
            logger.info(f"  Epoch {epoch} train loss: {train_loss}")
            # evaluate accuracy at end of each epoch
            Q_value.eval()

            test_loss = []
            for test_batch in test_buffer:
                value_loss = evaluate_model(test_batch).item()
                test_loss.append(value_loss)

            test_loss = np.mean(test_loss)
            logger.info(f"  Epoch {epoch} test loss: {test_loss}")

            history.append(test_loss)

            plt.clf()
            plt.plot(history)
            plt.savefig("value_fn_test_loss_curve.png")

            if test_loss < best_value_loss:
                logger.info(f"  Best value loss: {test_loss}")
                best_value_loss = test_loss
                best_weights = deepcopy(Q_value.state_dict())
                no_improvement = 0
            else:
                no_improvement += 1
            wandb.run.log(
                {"test_value_loss": test_loss, "train_value_loss": train_loss}
            )
            # if no_improvement > 500:
            # logger.info("  No improvement in 100 epochs. Stopping training.")
            # break

        model = Q_value
        # restore model and return best accuracy
        model.load_state_dict(best_weights)
        logger.info(f"Value Loss: {np.sqrt(best_value_loss)}")
        # plt.show()

        logger.info("  Evaluating value function")
        model.eval()
        # y_test = y_test.to("cpu")

        def plot_preds_vs_monte_carlo(buffer, suffix):
            episodes = buffer.episodes[:5]
            returns = np.array([traj["returns"] for traj in episodes])
            y_min, y_max = np.min(returns) - 1, np.max(returns) + 1
            with torch.no_grad():
                # Test out inference with 5 samples
                for traj_id, traj in enumerate(episodes):
                    states = traj["state"]
                    X_sample = scaler.transform(np.array(states))
                    X_sample = torch.tensor(X_sample, dtype=torch.float32).to(device)
                    G_sample = traj["returns"]
                    A_sample = torch.tensor(traj["action"], dtype=torch.int64).to(device)

                    if self.q_value:
                        y_pred = torch.gather(model(X_sample), dim=1,
                                              index=A_sample.unsqueeze(1)).cpu()
                    else:
                        y_pred = model(X_sample).cpu()
                    fig, ax = plt.subplots()
                    ax.set_ylim(y_min, y_max)
                    ax.plot(range(len(G_sample)), G_sample, label="monte carlo return")
                    ax.plot(
                        range(len(y_pred)),
                        y_pred.cpu().numpy().flatten(),
                        label="learned value function",
                    )
                    ax.legend()
                    wandb.run.log({f"value_fn_{suffix}_{traj_id}": fig})
                    fig.savefig(f"value_fn_{suffix}_{traj_id}.png")

        self.model = model

        torch.save(model, "value_fn.pt")
        pkl_dump(scaler, "value_fn_scaler.pkl")

        # plot_preds_vs_monte_carlo(train_buffer, "train")
        # plot_preds_vs_monte_carlo(test_buffer, "test")

        return model

    def load(self, path_to_model, path_to_scaler):
        self.model = torch.load(path_to_model, map_location=self.device)
        self.scaler = pkl_load(path_to_scaler)


def rews_to_returns(rews, discount=1):
    v = np.zeros_like(rews)
    for i, rew in enumerate(rews[::-1]):
        if i == 0:
            v[i] = rew
        else:
            v[i] = discount * v[i - 1] + rew
    return np.array(v[::-1])
