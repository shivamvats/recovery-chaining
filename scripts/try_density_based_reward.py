"""
Implements potential-based reward shaping where the poential function is
learned by density estimation of positive demos.
"""

import hydra
from hydra.utils import *
import numpy as np
from omegaconf import OmegaConf
import robosuite as rb

from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from hrl_manip.envs.improved_gym_wrapper import ImprovedGymWrapper
from hrl_manip.envs.time_wrapper import TimeWrapper
from hrl_manip.utils import *
from hrl_manip.envs.density_based_reward import DensityBasedReward
from hrl_manip.skills.nominal_skills import *


def make_env(cfg, env_cfg="default", eval=False):
    ctrl_cfg = OmegaConf.to_container(cfg.ctrl)
    env_cfg = OmegaConf.to_container(cfg.env)
    if cfg.failures.record or eval:
        env_cfg["has_offscreen_renderer"] = True
        env_cfg["use_camera_obs"] = True

    if cfg.task == "ShelfEnv":
        env_cfg["render_camera"] = "frontview"  # "sideview"
        # env_cfg["render_camera"] = "sideview"

    env_cfg["env_name"] = cfg.task
    env_cfg["has_renderer"] = cfg.render
    env_cfg["controller_configs"] = ctrl_cfg
    env_cfg["reward_cfg"] = cfg.rl.reward

    env = rb.make(env_configuration=env_cfg, **env_cfg)

    return env


obs_keys = ["box_pos", "box_mat", "shelf_pos", "robot0_gripper_pos", "robot0_eef_pos"]


def demos_to_states(demos):
    dataset = {"pos": [], "neg": []}
    for demo in demos["pos"]:
        obs_traj = [
            np.concatenate([state[key].flatten() for key in obs_keys]) for state in demo
        ]
        dataset["pos"].append(obs_traj)
    dataset["pos"] = np.concatenate(dataset["pos"])

    for demo in demos["neg"]:
        obs_traj = [
            np.concatenate([state[key].flatten() for key in obs_keys]) for state in demo
        ]
        dataset["neg"].append(obs_traj)
    dataset["neg"] = np.concatenate(dataset["neg"])

    return dataset


def compute_reward(potential_fn, pre_s, curr_s):
    # KDE-based reward estiamtion has no notion of "distance to goal".
    # It simply incentivies going to more highly occurring states.
    # Instead, I can fit a value function on the demos since I know action costs.
    # TODO Learn a value function from demos.
    pre_s = np.concatenate([pre_s[key].flatten() for key in obs_keys])
    curr_s = np.concatenate([curr_s[key].flatten() for key in obs_keys])
    pre_pot = potential_fn(np.array(pre_s).reshape(1, -1))
    curr_pot = potential_fn(np.array(curr_s).reshape(1, -1))
    rew = curr_pot - pre_pot
    return rew


def fit_value_fn(trajs):
    V_fn_keys = [
        "box_pos",
        "box_mat",
        "shelf_pos",
        "target_pos",
        "robot0_eef_pos",
        "robot0_gripper_pos",
    ]
    X, V = [], []

    def rews_to_values(rews):
        v = np.zeros_like(rews)
        for i, rew in enumerate(rews[::-1]):
            if i == 0:
                v[i] = rew
            else:
                v[i] = v[i - 1] + rew
        return v[::-1]

    for traj in trajs:
        obsx = [
            np.concatenate([transition[key].flatten() for key in V_fn_keys])
            for transition in traj["state"]
        ]
        rews = np.array(traj["reward"])
        values = rews_to_values(rews)
        X.append(obsx)
        V.append(values)

    X = np.concatenate(X, axis=0)
    V = np.concatenate(V)

    pkl_dump((X, V), "obs_value_fn.pkl")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, V, train_size=0.7, shuffle=True
    )

    # Standardizing data
    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    # training parameters
    n_epochs = 500  # number of epochs to run
    batch_size = 64  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)

    # Define the model
    model = nn.Sequential(
        nn.Linear(23, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )

    # loss function and optimizer
    loss_fn = nn.MSELoss()  # mean square error
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Hold the best model
    best_mse = np.inf  # init to infinity
    best_weights = None
    history = []

    print("Training")
    # training loop
    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start : start + batch_size]
                y_batch = y_train[start : start + batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                bar.set_postfix(mse=float(loss))
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_test)
        mse = loss_fn(y_pred, y_test)
        mse = float(mse)
        history.append(mse)
        if mse < best_mse:
            best_mse = mse
            best_weights = deepcopy(model.state_dict())

    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    print("RMSE: %.2f" % np.sqrt(best_mse))
    plt.plot(history)
    plt.savefig("value_fn_train_curve.png")
    # plt.show()

    print("Evaluating")
    model.eval()
    with torch.no_grad():
        # Test out inference with 5 samples
        for i in range(5):
            X_sample = X_test_raw[i : i + 1]
            X_sample = scaler.transform(X_sample)
            X_sample = torch.tensor(X_sample, dtype=torch.float32)
            y_pred = model(X_sample)
            print(
                f"{X_test_raw[i]} -> {y_pred[0].numpy()} (expected {y_test[i].numpy()})"
            )

    return model


def evaluate_reward(reward, cfg):
    pickup_skill = SidePickupSkill(target="box")
    goto_shelf_skill = GotoShelfSkill(target="target")
    place_skill = ShelfPlaceSkill(target="target")
    env = make_env(cfg)

    for _ in range(cfg.nevals):
        obs = env.reset()
        pre_s = env.observe_true_state(mj_state=False)
        obs, rew, done, info = pickup_skill.apply(env, obs, cfg.render)
        curr_s = env.observe_true_state(mj_state=False)
        rew = compute_reward(reward, pre_s, curr_s)
        print("Reward: ", rew)

        if not done:
            pre_s = env.observe_true_state(mj_state=False)
            obs, rew, done, info = goto_shelf_skill.apply(env, obs, cfg.render)
            curr_s = env.observe_true_state(mj_state=False)
            rew = compute_reward(reward, pre_s, curr_s)
            print("Reward: ", rew)

        if not done:
            pre_s = env.observe_true_state(mj_state=False)
            obs, rew, done, info = place_skill.apply(env, obs, cfg.render)
            curr_s = env.observe_true_state(mj_state=False)
            rew = compute_reward(reward, pre_s, curr_s)
            print("Reward: ", rew)


@hydra.main(config_path="../cfg", config_name="learn_recoveries.yaml")
def main(cfg):
    # demos = pkl_load(to_absolute_path("data/shelf/24-Aug/demos.pkl"))
    demos = pkl_load(to_absolute_path("data/shelf/29-Aug/demos.pkl"))
    fit_value_fn(demos["pos"] + demos["neg"])
    dataset = demos_to_states(demos)
    reward = DensityBasedReward(dataset, kernel_bandwidth="scott")
    evaluate_reward(reward, cfg)


if __name__ == "__main__":
    main()
