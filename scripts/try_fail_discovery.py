from os.path import join

import gym as gym
from gym.wrappers.normalize import NormalizeObservation
from argparse import ArgumentParser
import numpy as np
import d3rlpy as drl
from d3rlpy.online.buffers import ReplayBuffer
import matplotlib.pyplot as plt
import robosuite as rb
from robosuite.controllers import load_controller_config
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tqdm import tqdm

from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.model_selection import *

# from hrl_manip.gym_wrapper import GymWrapper
from hrl_manip.utils import *
from hrl_manip.envs.stack_v2 import StackV2
from hrl_manip.skills.primitive_skills import *


def make_env(args, cube_cfg="default"):
    ctrl_cfg = load_controller_config(default_controller="OSC_POSE")
    ctrl_cfg["control_delta"] = False
    env_cfg = {
        "reward_shaping": True,
        "horizon": 500,
        "has_renderer": args.render,
        "has_offscreen_renderer": False,
        "use_camera_obs": False,
        "controller_configs": ctrl_cfg,
    }
    # env = GymWrapper(
    # rb.make("StackV2", robots="Panda", env_configuration=cube_cfg, **env_cfg),
    # keys=["robot0_proprio-state", "object-state"],
    # )
    env = rb.make("StackV2", robots="Panda", env_configuration=cube_cfg, **env_cfg)

    return env


def make_sim_real_envs(args):
    sim_cubeA_size = 0.02
    sim_cubeB_size = 0.025

    real_cubeA_size = 0.02
    real_cubeB_size = 0.015

    sim_cfg = {
        "cubeA": {"size_min": [sim_cubeA_size] * 3, "size_max": [sim_cubeA_size] * 3},
        "cubeB": {"size_min": [sim_cubeB_size] * 3, "size_max": [sim_cubeB_size] * 3},
    }
    real_cfg = {
        "cubeA": {"size_min": [real_cubeA_size] * 3, "size_max": [real_cubeA_size] * 3},
        "cubeB": {"size_min": [real_cubeB_size] * 3, "size_max": [real_cubeB_size] * 3},
    }
    env_sim = make_env(args, sim_cfg)
    env_real = make_env(args, real_cfg)

    return env_sim, env_real


def get_nominal_skills(task="stacking"):
    if task == "stacking":
        pickup_skill = PickupSkill(target="cubeA")
        place_skill = PlaceSkill(target="cubeB")

        return pickup_skill, place_skill

    else:
        raise NotImplementedError


def evaluate_skills(env, skills, args):
    success = []
    for _ in tqdm(range(args.nevals)):
        obs = env.reset()
        for skill in skills:
            obs, rew, done, info = skill.apply(env, obs, render=args.render)
        success.append(env._check_success())
    return success


def plot_eef_forces(env, skills, args):
    for i in tqdm(range(args.nevals)):
        obs = env.reset()
        for skill in skills:
            obs, rew, done, info = skill.apply(env, obs, args.render)
        success = env._check_success()

        transitions = env.transitions
        ee_forces = []
        for s, a, s1 in transitions:
            ee_forces.append(s1["robot0_eef_forces"])
        ee_forces = np.array(ee_forces)

        fig, ax = plt.subplots()
        ax.plot(ee_forces[:, 0], label="x")
        ax.plot(ee_forces[:, 1], label="y")
        ax.plot(ee_forces[:, 2], label="z")
        ax.set_ylim([-30, 30])
        if success:
            ax.set_title("Success")
        else:
            ax.set_title("Failure")

        fig.legend()
        # plt.show()
        if success:
            plt.savefig(join(args.output_dir, f"succ_ee_forces_{i}.png"))
        else:
            plt.savefig(join(args.output_dir, f"fail_ee_forces_{i}.png"))


def state_transformer(obsx):
    """
    Transforms a state dict into an array for fitting classifier.
    """
    rel_keys = ["cubeA_pos", "cubeB_pos", "robot0_eef_pos", "robot0_eef_forces"]
    X = []
    for obs in obsx:
        x = np.concatenate([obs[key] for key in rel_keys])
        X.append(x)
    X = np.array(X)
    return X


def reset_env(env, state):
    """Reset env to a specific state."""
    env.sim.reset()
    env.sim.set_state_from_flattened(state["mj_state"])
    env.sim.forward()
    obs = env._get_observations(force_update=True)
    return obs


def train_failure_classifier(env, skills, args, one_class=False):
    """
    Trains a one-class SVM to detect OOD states.

    These states are classified as failures.
    """
    transitions_pos, transitions_neg = [], []
    for i in tqdm(range(args.nevals)):
        obs = env.reset()
        for skill in skills:
            obs, rew, done, info = skill.apply(env, obs, args.render)
        if env._check_success():
            transitions_pos.extend(env.transitions)
        else:
            transitions_neg.extend(env.transitions)
    states_pos = [t[-1] for t in transitions_pos[::2]]
    states_neg = [t[-1] for t in transitions_neg[::2]]

    if one_class:
        X_train, X_test = train_test_split(states_pos, test_size=0.2, random_state=42)
        pipe = Pipeline(
            [
                ("transformer", FunctionTransformer(state_transformer)),
                ("scaler", StandardScaler()),
                ("svm", OneClassSVM(kernel="rbf")),
            ]
        )
        param_grid = {"svm__nu": [0.1], "svm__gamma": [0.1]}
        grid_search = GridSearchCV(pipe, param_grid, scoring="accuracy", refit=True)
        grid_search.fit(X_train, y=None)
        best_params = grid_search.best_params_

        print("Grid search best params: ")
        print("-----------------------")
        print(best_params)

        clf = grid_search.best_estimator_

        print("Train classification report:")
        preds = clf.predict(X_train)
        y_train = np.ones(len(X_train))
        train_report = classification_report(y_train, preds)
        print(train_report)

        print("Test classification report:")
        preds = clf.predict(X_test)
        y_test = np.ones(len(X_test))
        test_report = classification_report(y_test, preds)
        print(test_report)

        print("Confusion matrix:")
        conf_mat = confusion_matrix(y_test, preds)
        print(conf_mat)

    else:

        from sklearn.neural_network import MLPClassifier

        X = np.concatenate([states_pos, states_neg])
        y = np.concatenate([np.ones(len(states_pos)), np.zeros(len(states_neg))])
        print("Data Stats:")
        print("  positive %: ", len(states_pos) / len(X) * 100)
        print("  negative %: ", len(states_neg) / len(X) * 100)

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2,
                                                            shuffle=True,
                                                            stratify=y,
                                                            random_state=42)
        pipe = Pipeline(
            [
                ("transformer", FunctionTransformer(state_transformer)),
                ("scaler", StandardScaler()),
                ("mlp", MLPClassifier(hidden_layer_sizes=(100, 100),
                                      random_state=1,
                                      max_iter=500))
            ]
        )
        param_grid = {"mlp__learning_rate_init": [0.0001, 0.0005, 0.001, 0.002]}
        grid_search = GridSearchCV(pipe, param_grid, scoring="balanced_accuracy", refit=True)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_

        print("Grid search best params: ")
        print("-----------------------")
        print(best_params)

        clf = grid_search.best_estimator_

        print("Train classification report:")
        preds = clf.predict(X_train)
        train_report = classification_report(y_train, preds)
        print(train_report)

        print("Test classification report:")
        preds = clf.predict(X_test)
        test_report = classification_report(y_test, preds)
        print(test_report)

        print("Confusion matrix:")
        conf_mat = confusion_matrix(y_test, preds)
        print(conf_mat)

    __import__('ipdb').set_trace()

    preds = clf.predict(states_neg)
    for pred, state in zip(preds, states_neg):
        if pred == -1:
            reset_env(env, state)
            for _ in range(25):
                env.render()


    clf_info = {
        "train_report": train_report,
        "test_report": test_report,
        "confusion_mat": conf_mat,
    }
    __import__('ipdb').set_trace()
    return clf, clf_info


def record_failures(env, skills, args):
    """
    A (state, action, state) is a failure if:
    1. it results in large contact forces
    2. task failure
    """
    fails = []
    for i in tqdm(range(args.nevals)):
        obs = env.reset()
        for skill in skills:
            obs, rew, done, info = skill.apply(env, obs, args.render)
        fails.extend(env.failures)

    return fails


def plot_failures(fails):
    eef_posx = np.array([fail[2]["robot0_eef_pos"] for fail in fails])
    cubeA_posx = np.array([fail[2]["cubeA_pos"] for fail in fails])
    cubeB_posx = np.array([fail[2]["cubeB_pos"] for fail in fails])
    print(eef_posx.shape)
    print(cubeA_posx.shape)
    print(cubeB_posx.shape)

    eef_to_cubeA = eef_posx - cubeA_posx
    eef_to_cubeB = eef_posx - cubeB_posx
    N = eef_to_cubeA.shape[0]

    fig, ax = plt.subplots()
    ax.scatter(np.arange(N), eef_to_cubeA[:, 0], label="x")
    ax.scatter(np.arange(N), eef_to_cubeA[:, 1], label="y")
    ax.scatter(np.arange(N), eef_to_cubeA[:, 2], label="z")
    ax.set_title("EEF to cubeA")
    fig.legend()
    plt.savefig(join(args.output_dir, f"eef_to_cubeA.png"))

    fig, ax = plt.subplots()
    ax.scatter(np.arange(N), eef_to_cubeB[:, 0], label="x")
    ax.scatter(np.arange(N), eef_to_cubeB[:, 1], label="y")
    ax.scatter(np.arange(N), eef_to_cubeB[:, 2], label="z")
    ax.set_title("EEF to cubeB")
    fig.legend()
    plt.savefig(join(args.output_dir, f"eef_to_cubeB.png"))


# Experiments
# -----------
def compare_perf_on_sim_real(env_sim, env_real, skills, seed):
    """
    Run the skills on sim and real envs and print success rate.
    """

    print("Evaluating on sim env:")
    set_seed(seed)
    success = evaluate_skills(env_sim, skills, args)
    print(f"  Success rate: {np.mean(success)*100}% ({np.sum(success)}/{len(success)})")

    print("\n")
    print("Evaluating on real env:")
    set_seed(seed)
    success = evaluate_skills(env_real, skills, args)
    print(f"  Success rate: {np.mean(success)*100}% ({np.sum(success)}/{len(success)})")


def main(args):
    env_sim, env_real = make_sim_real_envs(args)
    nom_skills = get_nominal_skills(task="stacking")

    # compare_perf_on_sim_real(env_sim, env_real, nom_skills, args.seed)
    compare_perf_on_sim_real(env_real, env_real, nom_skills, args.seed)
    # fails = record_failures(env_real, nom_skills, args)
    # plot_eef_forces(env_real, nom_skills, args)
    # fails = record_failures(env_real, nom_skills, args)
    # plot_failures(fails)

    # clf, clf_info = train_failure_classifier(env_real, nom_skills, args)

    env_sim.close()
    env_real.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp", type=str, default="")
    parser.add_argument("--output_dir", "-o", type=str, default="")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--nevals", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
