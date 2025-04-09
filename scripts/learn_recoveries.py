import logging
import os
from collections import OrderedDict
from copy import deepcopy
from os.path import join

import belief_srs.utils.transforms as T
import hydra
import matplotlib.pyplot as plt
import numpy as np
import robosuite as rb
import seaborn as sns
import torch
import wandb
from autolab_core import RigidTransform as RT
from belief_srs.envs.discrete_action_wrapper import DiscreteActionWrapper
from belief_srs.envs.improved_gym_wrapper import ImprovedGymWrapper
from belief_srs.envs.pick_env import PickBread
from belief_srs.envs.subtask_wrapper import SubtaskWrapper, reset_env
from belief_srs.envs.time_wrapper import TimeWrapper
from belief_srs.skills.nominal_skills import *
from belief_srs.skills.recovery_skill import RecoverySkillDRL
from belief_srs.skills.skill_chain import SkillChain
from belief_srs.skills.value_fn import NominalValueFunction
from belief_srs.utils import *
from hydra.utils import *
from omegaconf import OmegaConf
from PIL import Image
from robosuite.controllers import load_controller_config
from sklearn.model_selection import train_test_split
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (DummyVecEnv, VecNormalize,
                                              VecVideoRecorder)
from tqdm import tqdm

sns.set()


logger = logging.getLogger(__name__)
# logger.setLevel("INFO")
logger.setLevel("DEBUG")

device = torch.device("cpu")

INDICATOR_SITE_CONFIG = {
    "name": "indicator0",
    "type": "sphere",
    "size": [0.01],
    "rgba": [1, 0, 0, 0.5],
}


def make_env(cfg, env_cfg="default", test=False):
    ctrl_cfg = OmegaConf.to_container(cfg.ctrl)
    env_cfg = OmegaConf.to_container(cfg.env)
    # if cfg.failures.record or test:
    if cfg.evaluate.record_video or cfg.failures.record:
        env_cfg["has_offscreen_renderer"] = True
        env_cfg["use_camera_obs"] = True

    if env_cfg["env_name"] == "ShelfEnv" or env_cfg["env_name"] == "ShelfEnvReal":
        env_cfg["render_camera"] = "frontview"  # "sideview"
        env_cfg["reward_cfg"] = cfg.rl.reward
        if "failure_detection" in cfg.rl:
            env_cfg["failure_detection"] = cfg.rl.failure_detection
        # env_cfg["render_camera"] = "sideview"

    # env_cfg["env_name"] = cfg.env.env_name
    env_cfg["has_renderer"] = cfg.render
    env_cfg["controller_configs"] = ctrl_cfg
    if cfg.env.reward_shaping and cfg.rl.reward.potential.enabled:
        # if os.path.isfile(cfg.value_fn.model_filename):
        # value_fn = load_value_fn(cfg.value_fn, device="cpu")
        # value_fn.model.to(torch.device("cpu"))
        # env_cfg["potential_fn"] = value_fn
        # restore model checkpoint
        q_value = wandb.restore(
            "q_value.d3", run_path=cfg.q_value.model_run_path)
        # now the model is in my local dir
        cfg.q_value.model_filename = q_value.name
        q_value = d3.load_learnable(q_value.name)

        def predict_value(obsx):
            obs_arr = np.array(
                [
                    np.concatenate(
                        [
                            np.array(obs[key]).flatten()
                            for key in cfg.q_value.obs_keys
                            if key in obs
                        ]
                    )
                    for obs in obsx
                ]
            )
            actions = 2 * np.ones(len(obsx))
            return q_value.predict_value(obs_arr, actions)

        env_cfg["potential_fn"] = predict_value

    # env = rb.make(env_configuration=env_cfg, **env_cfg)
    env = rb.make(**env_cfg)

    if cfg.env.env_name == "PickPlaceBread":
        from belief_srs.envs.failure_detector_wrapper import \
            PickPlaceFailureDetector

        failure_detection = cfg.rl.get("failure_detection", True)
        env = PickPlaceFailureDetector(
            env, cfg, failure_detection=failure_detection)

    elif cfg.env.env_name == "PickBread":
        from belief_srs.envs.failure_detector_wrapper import \
            PickFailureDetector

        failure_detection = cfg.rl.get("failure_detection", True)
        env = PickFailureDetector(
            env, cfg, failure_detection=failure_detection)


    return env


def wrap_env(
    base_env,
    cfg,
    start_states,
    subgoal,
    cluster_key,
    render_mode="rgb_array",
    test=False,
):
    env = TimeWrapper(base_env)
    rl_cfg = deepcopy(cfg.rl)
    if test:
        rl_cfg.use_nominal_reward_model = False

    if rl_cfg.use_discrete_actions:
        if rl_cfg.use_nominal_reward_model or test:
            # also needed for picking which nominal skill to exec in skill chaining

            if rl_cfg.get("use_monte_carlo_q", False):
                if os.path.isfile(cfg.value_fn.model_filename):
                    value_fn = load_value_fn(cfg.value_fn, device="cpu")
                    value_fn.model.to(torch.device("cpu"))
                    nom_reward_model = value_fn
                else:
                    raise FileNotFoundError("Value function file not found.")

            elif rl_cfg.get("use_cql_q", False):
                # d3rlpy
                # --------
                q_value = wandb.restore(
                    "q_value.d3", run_path=cfg.q_value.model_run_path
                )
                # now the model is in my local dir
                cfg.q_value.model_filename = q_value.name
                q_value = d3.load_learnable(q_value.name)

                def nom_reward_model(obsx, actions):
                    obs_arr = np.array(
                        [
                            np.concatenate(
                                [
                                    np.array(obs[key]).flatten()
                                    for key in rl_cfg.obs_keys
                                ]
                            )
                            for obs in obsx
                        ]
                    )
                    return q_value.predict_value(obs_arr, np.array(actions))

            else:
                nom_reward_model = None

        else:
            nom_reward_model = None

        evaluate_skill_chain = rl_cfg.get("evaluate_skill_chain", False)
        if not test:
            # only True for skill chaining and in test
            evaluate_skill_chain = False

        env = DiscreteActionWrapper(
            env, rl_cfg, nom_reward_model, evaluate_skill_chain=evaluate_skill_chain
        )

        if rl_cfg.algorithm == "PPOTermQ" and not test:
            env.set_execute_nominal_actions(False)

        nom_skills = get_nominal_skills(cfg.env.env_name)
        if rl_cfg.use_nom_actions:
            if rl_cfg.use_nom_actions == "all":
                for target_precond in range(len(nom_skills)):
                    nom_skill = SkillChain(nom_skills[target_precond:])
                    logger.info(
                        f"  Loading skill chain with skills: {nom_skill.skills}"
                    )
                    env.add_action(nom_skill, term_action=True)

            elif rl_cfg.use_nom_actions == "prim":
                for nom_skill in nom_skills:
                    logger.info(
                        f"  Loading skill as a primitive action: {nom_skill}")
                    env.add_action(nom_skill, term_action=False)

            else:
                target_precond = cluster_key[0] + 1
                nom_skill = SkillChain(nom_skills[target_precond:])
                logger.info(
                    f"  Loading skill chain with skills: {nom_skill.skills}")
                env.add_action(nom_skill, term_action=True)

        if test:
            nom_chains = [
                SkillChain(nom_skills[target:]) for target in range(len(nom_skills))
            ]
            env.set_nominal_chains_for_eval(nom_chains)

    env = SubtaskWrapper(env, start_states=start_states, subgoal=subgoal)
    if "subtask_reset" in rl_cfg:
        env.subtask_reset = rl_cfg.subtask_reset
        logger.info(f"  Setting subtask reset: {env.subtask_reset}")

    if cfg.rl.use_skill_actions:
        # TODO
        raise NotImplementedError("Skill actions not yet implemented.")

    if cfg.rl.use_recovery_actions:
        skill_env = ImprovedGymWrapper(env, keys=rl_cfg.obs_keys)
        if cfg.render:
            skill_env.render_mode = render_mode
        skill_env = DummyVecEnv([lambda: skill_env])
        for skill_name, skill_path in skill_paths.items():
            logger.info(f"  Adding skill {skill_name} as action")
            skill = RecoverySkillDRL(cfg)
            skill.load(
                to_absolute_path(skill_path["path_to_policy"]),
                to_absolute_path(skill_path["path_to_vnorm"]),
                skill_env,
                device=device,
            )
            env.add_action(skill, cost=1, term_action=False)

    env = ImprovedGymWrapper(env, keys=rl_cfg.obs_keys)
    if cfg.render:
        env.render_mode = render_mode

    return env


def make_sim_real_envs(cfg):
    if cfg.env.env_name == "Stack":
        sim_cubeA_size = 0.02
        sim_cubeB_size = 0.025

        real_cubeA_size = 0.02
        real_cubeB_size = 0.015

        sim_cfg = {
            "cubeA": {
                "size_min": [sim_cubeA_size] * 3,
                "size_max": [sim_cubeA_size] * 3,
            },
            "cubeB": {
                "size_min": [sim_cubeB_size] * 3,
                "size_max": [sim_cubeB_size] * 3,
            },
        }
        real_cfg = {
            "cubeA": {
                "size_min": [real_cubeA_size] * 3,
                "size_max": [real_cubeA_size] * 3,
            },
            "cubeB": {
                "size_min": [real_cubeB_size] * 3,
                "size_max": [real_cubeB_size] * 3,
            },
        }
        env_sim = make_env(cfg, sim_cfg)
        env_real = make_env(cfg, real_cfg)

    else:
        env_sim = make_env(cfg)
        env_real = make_env(cfg)

    return env_sim, env_real


def get_nominal_skills(task="Stack"):
    if task == "Stack":
        pickup_skill = PickupSkill(target="cubeA")
        place_skill = BlockPlaceSkill(target="cubeB")

        return pickup_skill, place_skill

    elif task == "PlaceAtEdge":
        pickup_skill = PickupSkill(target="cube")
        place_skill = TablePlaceSkill(target="target")

        return pickup_skill, place_skill

    elif task == "ShelfEnv":
        pickup_skill = SideGraspSkill(target="box")
        goto_shelf_skill = GotoShelfSkill(target="target")
        place_skill = ShelfPlaceSkill(target="target")

        return pickup_skill, goto_shelf_skill, place_skill

    elif task == "PickPlaceBread":
        goto_grasp_skill = GotoGraspSkill(target="Bread")
        pickup_skill = PickupSkill(target="Bread")
        goto_goal_skill = GotoGoalSkill("Bread")
        place_skill = PlaceSkill("Bread")

        return goto_grasp_skill, pickup_skill, goto_goal_skill, place_skill

    elif task == "Door":
        from belief_srs.skills.door_nominal_skills import (
            PullHandleSkill, ReachAndGraspHandleSkill, RotateHandleSkill)

        goto_skill = ReachAndGraspHandleSkill()
        rotate_skill = RotateHandleSkill()
        pull_skill = PullHandleSkill()

        # return goto_skill, rotate_skill, pull_skill
        return goto_skill, pull_skill

    elif task == "PickBread":
        pickup_skill = SinglePickupSkill(target="Bread")

        return (pickup_skill,)

    else:
        raise NotImplementedError


def collect_failures(env, skills, cfg):
    env = TimeWrapper(env)

    succs, rews = [], []
    n_fails, n_trans = 0, 0
    pbar = tqdm(total=cfg.failures.nfails)
    failures, demos = [], []
    while n_fails < cfg.failures.nfails and len(succs) < cfg.failures.max_evals:
        logger.info(f"  n_fails: {n_fails}, n_executions: {len(demos)}")
        total_rew = 0
        obs = env.reset()
        traj = {"state": [], "reward": [], "action": [], "ll_action": []}
        traj["state"].append(env.observe_true_state(mj_state=False))
        for skill in skills:
            # logger.info(f"  executing {skill}")
            obs, rew, done, info = skill.apply(env, obs, render=cfg.render)
            total_rew += rew
            traj["state"].extend(info["hist"]["state"])
            traj["reward"].extend(info["hist"]["reward"])
            traj["action"].extend(info["hist"]["skill"])
            if "action" in info["hist"]:
                traj["ll_action"].extend(info["hist"]["action"])
            if done:
                break

        is_failure = info["is_failure"]
        demos.append(traj)
        n_trans += len(traj["reward"])
        if is_failure:
            failures.append(info["state"])
            n_fails += 1
        success = total_rew > 0

        logger.info(
            f"Success: {success}, term: {is_failure}, total reward: {total_rew}, traj_len: {len(traj['reward'])}, #transitions: {n_trans}"
        )
        rews.append(rew)
        succs.append(success)
    pbar.close()

    pkl_dump(failures, "failures.pkl")
    pkl_dump(demos, "demos.pkl")
    logger.info("Statistics:")
    logger.info("-------------")
    logger.info(f"  #evals: {len(demos)}")
    logger.info(f"  Reward:  {np.mean(rews)} +- {np.std(rews)}")
    logger.info(f"  Success rate : {np.mean(succs) * 100}")
    return failures, demos


def fit_value_fn(trajs, cfg):
    """Fit a value function on trajectories"""
    logger.info("  Fitting nominal value function")
    value_fn = NominalValueFunction(cfg)
    value_fn.train(trajs)
    return value_fn


def load_value_fn(cfg, device="cpu"):
    try:
        value_fn = NominalValueFunction(cfg, device)
        value_fn.load(cfg.model_filename, cfg.scaler_filename)
    except:
        pass

    return value_fn


def cluster_failures(fails, cfg):
    fail_types = ["collision", "slip", "collision-slip", "missed_obj"]
    stages = [0, 1, 2]

    all_states = OrderedDict()
    if cfg.clusters.cluster_by == "None":
        all_states[(1,)] = []
    elif cfg.clusters.cluster_by == "stage":
        for stage in stages:
            key = (stage,)
            all_states[key] = []

    elif cfg.clusters.cluster_by == "stage+fail_type":
        for stage in stages:
            for fail_type in fail_types:
                key = (stage, fail_type)
                all_states[key] = []
    else:
        raise NotImplementedError

    for fail in fails:
        if cfg.clusters.cluster_by == "None":
            key = (1,)
        elif cfg.clusters.cluster_by == "stage":
            key = (fail["stage"],)
        elif cfg.clusters.cluster_by == "stage+fail_type":
            key = (fail["stage"], fail["type"])
        else:
            raise NotImplementedError
        all_states[key].append(fail)

    logger.info("  Failure clusters:")
    logger.info("-------------")
    for key in all_states.keys():
        logger.info(f"    {key}: {len(all_states[key])}")

    all_states = all_states

    clusters = {
        "train": {key: [] for key in all_states.keys()},
        "test": {key: [] for key in all_states.keys()},
    }

    for c_key, c_states in all_states.items():
        if len(c_states) > 5:
            train_states, test_states = train_test_split(
                c_states, shuffle=True, test_size=0.2
            )
        else:
            train_states, test_states = c_states, []
        clusters["train"][c_key] = train_states
        clusters["test"][c_key] = test_states

    pkl_dump(clusters, "fail_clusters.pkl")
    return clusters


def learn_recoveries(init_failures, cfg, warmstart=None):
    recoveries = []

    for key in cfg.recoveries.clusters:
        key = tuple(key)
        fail_cluster = init_failures[key]
        # fail_cluster = clusters[key][:1]
        try:
            cluster = [fail["state"] for fail in fail_cluster]
        except KeyError:
            cluster = fail_cluster

        if cfg.env.env_name == "PlaceAtEdge":
            params = np.array([0.0, -0.02, 0.0, 1])
            recovery = BlockRecoverySkill(params)

            def train_env_fn():
                env = make_env(cfg)
                env = SubtaskWrapper(env, start_states=cluster, subgoal=None)
                return env

            recovery.train_policy(train_env_fn, train_env_fn, cfg)
            recoveries.append(recovery)

        else:
            params = None
            recovery = RecoverySkillDRL(cfg)

            def env_fn(cfg, render_mode, test, render=False):
                cfg = deepcopy(cfg)
                if render:
                    cfg.render = True
                base_env = make_env(cfg, test=test)
                env = wrap_env(
                    base_env,
                    cfg,
                    cluster,
                    cfg.recoveries.goal,
                    key,
                    render_mode,
                    test,
                )
                return env

            recovery.train_policy(
                lambda: env_fn(cfg, "human", test=False),
                lambda: env_fn(cfg, "rgb_array", test=True),
            )
            recoveries.append(recovery)

        pkl_dump(recoveries, "recoveries.pkl")

    return recoveries


def evaluate_recoveries(
    clusters,
    env,
    cfg,
    eval_from_start=False,
    nom_skills=None,
):
    subgoal = None  # task reward

    recovery = RecoverySkillDRL(cfg)
    logger.info(
        f"  Loading recovery policy from {cfg.evaluate.path_to_policy}")
    logger.info(f"  Loading recovery vnorm from {cfg.evaluate.path_to_vnorm}")
    if not eval_from_start:
        for key in cfg.recoveries.clusters:
            key = tuple(key)
            fail_cluster = clusters[key]
            if "state" in fail_cluster[0]:
                cluster = [fail["state"] for fail in fail_cluster]
            else:
                cluster = fail_cluster

            base_env = env
            env = wrap_env(
                env,
                cfg,
                cluster,
                cfg.recoveries.goal,
                key,
                render_mode="human",
                test=True,
            )
            # env = VisualizationWrapper(env, indicator_configs=INDICATOR_SITE_CONFIG)

            env = DummyVecEnv([lambda: env])
            if cfg.evaluate.record_video and not cfg.render:
                env = VecVideoRecorder(
                    env,
                    "./videos/",
                    record_video_trigger=lambda x: True,
                    video_length=1024,
                    name_prefix="eval_video",
                )

            recovery.load(
                to_absolute_path(cfg.evaluate.path_to_policy),
                to_absolute_path(cfg.evaluate.path_to_vnorm),
                env,
            )

            eval_info = {"rews": [], "succs": [],
                         "starts": [], "term_states": []}
            for _ in tqdm(range(cfg.nevals)):
                obs = env.reset()
                if cfg.render:
                    for _ in range(5):
                        env.render()
                start_state = env.envs[0].observe_true_state()
                eval_info["starts"].append(start_state)
                obs, rew, done, info = recovery.apply(obs)
                logger.info(f"  rew: {rew}, success: {info['is_success']}")
                eval_info["term_states"].append(
                    env.envs[0].observe_true_state())
                eval_info["rews"].append(rew)
                eval_info["succs"].append(info["is_success"])

            pkl_dump(eval_info, "eval_info.pkl")

    else:
        eval_info = {
            "nom_succs": [],
            "rews": [],
            "succs": [],
            "recovery_succs": [],
            "starts": [],
            "term_states": [],
        }
        for key in cfg.recoveries.clusters:
            key = tuple(key)
            fail_cluster = clusters[key]
            if "state" in fail_cluster[0]:
                cluster = [fail["state"] for fail in fail_cluster]
            else:
                cluster = fail_cluster
            base_env = env

            env = wrap_env(
                env,
                cfg,
                cluster,
                cfg.recoveries.goal,
                key,
                render_mode="human",
                test=True,
            )
            env.set_subtask_reset(False)
            logger.info("  Set subtask reset False")
            env = DummyVecEnv([lambda: env])

            recovery.load(
                to_absolute_path(cfg.evaluate.path_to_policy),
                to_absolute_path(cfg.evaluate.path_to_vnorm),
                env,
            )

            import imageio.v2 as iio

            n_fails = 0
            for i in tqdm(range(cfg.nevals)):
                total_rew = 0

                if cfg.evaluate.record_video:
                    writer = iio.get_writer(f"eval_video_{i}.mp4", fps=20)

                obs = env.reset()

                if nom_skills is not None:
                    obs = base_env._get_observations()
                    logger.debug("  executing nominal")

                    for skill in nom_skills:
                        # logger.info(f"  executing {skill}")
                        obs, rew, done, info = skill.apply(
                            base_env, obs, render=cfg.render
                        )
                        if cfg.evaluate.record_video:
                            for state in info["hist"]["state"]:
                                writer.append_data(
                                    state["obs"][f"{cfg.env.camera_names}_image"][::-1]
                                )
                        total_rew += rew
                        if done:
                            break
                    is_failure = info["is_failure"]

                    if info["is_success"]:
                        eval_info["nom_succs"].append(1)
                    else:
                        eval_info["nom_succs"].append(0)

                    if is_failure:
                        logger.debug("  failure. executing recovery")
                        n_fails += 1
                        # trigger recovery
                        # reset_failure = True
                        reset_failure = False
                        if reset_failure:
                            state = env.envs[0].observe_true_state(
                                mj_state=True)
                            obs = reset_env(env.envs[0], state).reshape(1, -1)
                        else:
                            obs = env.envs[0]._get_observations().reshape(
                                1, -1)
                        obs, rew, done, info = recovery.apply(obs)
                        if cfg.evaluate.record_video:
                            for state in info["hist"]["state"]:
                                writer.append_data(
                                    state["obs"][f"{cfg.env.camera_names}_image"][::-1]
                                )

                        if info["is_success"]:
                            logger.debug("    recovery success")
                            eval_info["recovery_succs"].append(True)
                        else:
                            logger.debug("    recovery failed")
                            eval_info["recovery_succs"].append(False)
                        if cfg.evaluate.record_video:
                            writer.close()

                else:
                    obs, rew, done, info = recovery.apply(obs)

                    if info["is_success"]:
                        logger.debug("    recovery success")
                    else:
                        logger.debug("    recovery failed")

                logger.debug(f"  rew: {rew}, success: {info['is_success']}")

                eval_info["rews"].append(rew)
                eval_info["succs"].append(info["is_success"])

                pkl_dump(eval_info, "eval_info.pkl")

        logger.info("Evaluation stats:")
        logger.info("------------------")
        logger.info(f" Reward: mean: {np.mean(eval_info['rews'])}")
        logger.info(f" Success rate: {np.mean(eval_info['succs'])*100}%")
        logger.info(
            f" Nominal success rate: {np.mean(eval_info['nom_succs'])*100}%")
        logger.info(
            f" Recovery rate: {np.mean(eval_info['recovery_succs'])*100}%")

        # visualize precondition
        # start_states = np.array([start["obs"]["slip"] for start in eval_info["starts"]])
        # y = np.array(eval_info["succs"])
        # plt.scatter(
        # start_states[y == 0],
        # np.zeros(len(start_states[y == 0])),
        # color="r",
        # label="Fail",
        # )
        # plt.scatter(
        # start_states[y == 1],
        # np.zeros(len(start_states[y == 1])),
        # color="g",
        # label="Success",
        # )
        # plt.title("Slip vs success")
        # plt.savefig("slip_vs_success.png")

    return eval_info


def compute_init_set(recovery, fail_cluster, env, cfg):
    eval_info = {"rews": [], "succs": [], "starts": [], "term_states": []}

    for i, fail in enumerate(tqdm(fail_cluster)):
        state = fail["state"]
        env.envs[0].set_start_states([state])
        eval_info["starts"].append(env.envs[0].observe_true_state())
        succs = []
        for _ in range(cfg.recoveries.n_robust_evaluations):
            # resample uncertainty
            obs = env.reset()
            if cfg.render:
                for _ in range(25):
                    env.render()
            # env.envs[0].set_indicator_pos("indicator0", env.envs[0]._target_pos)
            obs, rew, done, info = recovery.apply(obs)
            eval_info["term_states"].append(env.envs[0].observe_true_state())
            eval_info["rews"].append(rew)
            succs.append(info["is_success"])
        precond = np.mean(succs)
        logger.info(
            f"    Failure {i} precondition: {precond*100}% ({np.sum(succs)}/{len(succs)})"
        )
        eval_info["succs"].append(precond)

    pkl_dump(eval_info, "eval_info.pkl")

    logger.info("Evaluation stats:")
    logger.info("------------------")
    logger.info(f"  Reward: mean: {np.mean(eval_info['rews'])}")
    logger.info(f"  Success rate: {np.mean(eval_info['succs'])*100}%")

    preconds = np.array(eval_info["succs"])
    thresh = cfg.cascaded.precond_thresh
    logger.info(
        f"  Init set coverage (thresh={thresh}): {np.mean(preconds >= thresh)*100}%"
    )
    init_set = np.array(fail_cluster)[preconds >= thresh]
    unsolved_states = np.array(fail_cluster)[preconds < thresh]
    pkl_dump(init_set, "solved_fails.pkl")
    pkl_dump(unsolved_states, "unsolved_fails.pkl")
    pkl_dump(eval_info, "init_set_eval_info.pkl")

    return init_set, unsolved_states, eval_info


def evaluate_recovery_chain(base_env, cfg):
    recovery = RecoverySkillDRL(cfg)
    pick_skill, goto_skill, place_skill = get_nominal_skills(
        task=cfg.env.env_name)

    skill_paths = {}
    for skill_name, skill_path in cfg.rl.skills.items():
        skill_paths[skill_name] = {
            "path_to_policy": to_absolute_path(skill_path["path_to_policy"]),
            "path_to_vnorm": to_absolute_path(skill_path["path_to_vnorm"]),
        }

    def gym_env_fn(env, render_mode="human"):
        if cfg.rl.use_discrete_actions:
            env = DiscreteActionWrapper(env, cfg.rl)
            # env = TimeWrapper(env, cfg.rl.hl_horizon)
            nom_skill = ShelfPlaceSkill(target="target")
            env.add_action(nom_skill, cost=1)

        if cfg.rl.use_recovery_actions:
            # skill_env = TimeWrapper(env, cfg.rl.hl_horizon)
            skill_env = ImprovedGymWrapper(skill_env, keys=cfg.rl.obs_keys)
            if cfg.render:
                skill_env.render_mode = render_mode
            skill_env = DummyVecEnv([lambda: skill_env])
            for skill_name, skill_path in skill_paths.items():
                logger.info(f"  Adding skill {skill_name} as action")
                skill = RecoverySkillDRL(cfg)
                skill.load(
                    to_absolute_path(skill_path["path_to_policy"]),
                    to_absolute_path(skill_path["path_to_vnorm"]),
                    skill_env,
                    device=device,
                )
                env.add_action(skill, cost=1, term_action=False)

        env = ImprovedGymWrapper(env, keys=cfg.rl.obs_keys)
        if cfg.render:
            env.render_mode = render_mode

        return env

    # base_env = VisualizationWrapper(base_env, indicator_configs=INDICATOR_SITE_CONFIG)

    gym_env = gym_env_fn(base_env)
    gym_env = DummyVecEnv([lambda: gym_env])

    recovery.load(
        to_absolute_path(cfg.evaluate.path_to_policy),
        to_absolute_path(cfg.evaluate.path_to_vnorm),
        gym_env,
    )

    eval_info = {"rews": [], "succs": [], "starts": [], "term_states": []}
    n_fails = 0
    for _ in tqdm(range(cfg.nevals)):
        FAIL = False
        obs = base_env.reset()
        start_state = base_env.observe_true_state()
        eval_info["starts"].append(start_state)
        # base_env.set_indicator_pos("indicator0", base_env._target_pos)

        base_env.monitor_failure = {
            "collision": True,
            "slip": False,
            "missed_obj": False,
        }
        base_env.set_stage(0)
        obs, rew, done, info = pick_skill.apply(
            base_env, obs, render=cfg.render)

        if not done:
            base_env.monitor_failure = {
                "collision": True,
                "slip": False,
                "missed_obj": False,
            }
            base_env.set_stage(1)
            obs, rew, done, info = goto_skill.apply(
                base_env, obs, render=cfg.render)

        if len(base_env.failures) > n_fails:
            n_fails = len(base_env.failures)

        if not done:
            base_env.set_stage(1)
            obs, rew, done, info = place_skill.apply(
                base_env, obs, render=cfg.render)

        if base_env._check_terminated():
            FAIL = True

        if FAIL:
            logger.info("  Executing recovery")
            for _ in range(50):
                base_env.render()
            base_env.monitor_failure = {
                "collision": True,
                "slip": False,
                "missed_obj": False,
            }
            obs = gym_env.envs[0]._get_observations().reshape(1, -1)
            # reset option time
            obs, rew, done, info = recovery.apply(obs)

        logger.info(f"  rew: {rew}, success: {info['is_success']}")
        eval_info["term_states"].append(base_env.observe_true_state())
        eval_info["rews"].append(rew)
        eval_info["succs"].append(info["is_success"])

    pkl_dump(eval_info, "eval_info.pkl")

    logger.info("Evaluation stats:")
    logger.info("------------------")
    logger.info(f" Reward: mean: {np.mean(eval_info['rews'])}")
    logger.info(f" Success rate: {np.mean(eval_info['succs'])*100}%")

    return eval_info


def viz_states(env, states):
    for i, state in enumerate(states):
        obs = reset_env(env, state)
        for _ in range(50):
            env.sim.forward()
            env.render()


def record_states(env, states):
    for i, state in enumerate(states):
        obs = reset_env(env, state)
        img = Image.fromarray(obs[f"{env.render_camera}_image"][::-1])
        img.save(f"state_{i}.png")


def record_nominal_traj(nom_skills, n_trajs, cfg):
    """
    Rollout nominal policies and save the states on the trajectory.
    """

    cfg = deepcopy(cfg)
    cfg["env"]["camera_heights"] = 2048
    cfg["env"]["camera_widths"] = 2048
    cfg["env"]["has_offscreen_renderer"] = True
    cfg["env"]["use_camera_obs"] = True
    env = make_env(cfg, test=True)

    obs = env.reset()
    # execute skills on the same task to get different images states
    state = env.observe_true_state()

    for i in range(n_trajs):
        obs = reset_env(env, state)

        j = 0
        img = Image.fromarray(obs[f"{env.render_camera}_image"][::-1])
        img.save(f"traj_{i}_state_{j}.png")

        for skill in nom_skills:
            j += 1
            obs, rew, done, info = skill.apply(env, obs)
            img = Image.fromarray(obs[f"{env.render_camera}_image"][::-1])
            img.save(f"traj_{i}_state_{j}.png")
            if done:
                break

    logger.info(f"Saving images from {n_trajs} trajectories.")


# --------
# Evaluation methods
# ---------
def evaluate_nominal_skills(env, skills, cfg):
    env = TimeWrapper(env)
    if cfg.env.env_name == "ShelfEnv":
        pick_skill, goto_skill, place_skill = skills

        skill_success = {"pick": [], "goto": [], "place": []}
        succs, rews = [], []
        n_fails = 0
        pbar = tqdm(total=cfg.failures.nfails)
        for i in range(cfg.nevals):
            total_rew = 0

            obs = env.reset()
            obs, rew, done, info = pick_skill.apply(
                env, obs, render=cfg.render)
            total_rew += rew
            if not done:
                obs, rew, done, info = goto_skill.apply(
                    env, obs, render=cfg.render)
                eef_end = obs["robot0_eef_pos"]
                total_rew += rew

            if not done:
                obs, rew, done, info = place_skill.apply(
                    env, obs, render=cfg.render)
                total_rew += rew

            if env._check_terminated():
                # pbar.update(len(env.failures) - n_fails)
                failure = env.failure
                # traj state has option_time
                failure["state"] = deepcopy(traj["state"][-1])
                failures.append(failure)
                n_fails += 1

            success = env._check_success()
            if success:
                demos["pos"].append(traj)
            else:
                demos["neg"].append(traj)

            print(
                f"Success: {success}, term: {env._check_terminated()} total reward: {total_rew}"
            )
            rews.append(rew)
            succs.append(success)
        pbar.close()

    elif cfg.env.env_name == "PickPlaceCan":
        (pick_skill,) = skills

        succs, rews = [], []
        n_fails = 0
        pbar = tqdm(total=cfg.failures.nfails)
        demos = {"pos": [], "neg": []}
        failures = []
        while n_fails < cfg.failures.nfails and len(succs) < cfg.failures.max_evals:
            print(
                f"n_fails: {n_fails}, trajs: {len(demos['pos']) + len(demos['neg'])}")
            total_rew = 0
            traj = {"state": [], "reward": [], "action": []}

            obs = env.reset()
            # false positives on collision during pickup
            # let objects settle
            for _ in range(5):
                env.sim.forward()
                env.sim.step()
                if cfg.render:
                    env.render()

            env.monitor_failure = {"collision": False,
                                   "slip": True, "missed_obj": True}
            # env.set_stage(0)
            # true_state = env.observe_true_state(mj_state=True)
            # true_state["option_time"] = np.array([0])
            # traj["state"].append(true_state)
            # traj["state"].append(env.observe_true_state(mj_state=True))
            eef_start = obs["robot0_eef_pos"]
            obs, rew, done, info = pick_skill.apply(
                env, obs, render=cfg.render)
            eef_end = obs["robot0_eef_pos"]
            total_rew += rew
            traj["state"].extend(info["hist"]["state"])
            traj["reward"].extend(info["hist"]["reward"])
            # traj["action"].extend(
            # env.stage * np.ones(len(info["hist"]["reward"]), dtype=int)
            # )

            # if env._check_terminated():
            if False:
                # pbar.update(len(env.failures) - n_fails)
                failure = env.failure
                # traj state has option_time
                failure["state"] = deepcopy(traj["state"][-1])
                failures.append(failure)
                n_fails += 1

            success = env._check_success()
            if success:
                demos["pos"].append(traj)
            else:
                demos["neg"].append(traj)

            print(
                # f"Success: {success}, term: {env._check_terminated()} total reward: {total_rew}"
                f"Success: {success}, term: False, total reward: {total_rew}"
            )
            rews.append(rew)
            succs.append(success)
            if cfg.failures.debug:
                plt.plot(np.max(failures, axis=1))
                plt.title("EEF Forces")
                plt.show()
        pbar.close()

    elif cfg.env.env_name == "PlaceAtEdge":
        pick_skill, place_skill = skills
        rews = []
        n_fails = 0
        pbar = tqdm(total=cfg.failures.nfails)
        while n_fails < cfg.failures.nfails:
            obs = env.reset()
            obs, rew, done, info = pick_skill.apply(
                env, obs, render=cfg.render)
            # env.monitor_failure = True
            obs, rew, done, info = place_skill.apply(
                env, obs, render=cfg.render)
            # env.monitor_failure = False
            if len(env.failures) > n_fails:
                pbar.update(len(env.failures) - n_fails)
                n_fails = len(env.failures)

            logger.debug(f"Success: {env._check_success()}")
            logger.debug(f"Reward: {rew}")
            rews.append(rew)
        pbar.close()

    else:
        raise NotImplementedError

    fails = failures
    pkl_dump(fails, "failures.pkl")
    pkl_dump(demos, "demos.pkl")
    logger.info(f"Mean rew: {np.mean(rews)}")
    logger.info(f"Success rate : {np.mean(succs) * 100}")
    return fails, demos


@hydra.main(config_path="../cfg", config_name="learn_recoveries.yaml")
def main(cfg):
    wandb.require("service")
    if cfg.env.env_name == "ShelfEnv":
        if cfg.env.task_dist.clutter.enable:
            cfg.data_dir = cfg.shelf_clutter_data_dir
            cfg.value_fn.obs_keys = cfg.value_fn.cluttered_shelf_obs_keys
        else:
            cfg.data_dir = cfg.shelf_data_dir
            cfg.value_fn.obs_keys = cfg.value_fn.shelf_obs_keys
    elif cfg.env.env_name == "PickPlaceBread":
        cfg.data_dir = cfg.pick_place_data_dir
    elif cfg.env.env_name == "Door":
        cfg.data_dir = cfg.door_data_dir
    elif cfg.env.env_name == "PickBread":
        cfg.data_dir = cfg.pick_data_dir

    cfg.output_dir = os.getcwd()
    set_random_seed(cfg.seed)
    if cfg.use_wandb:
        config = OmegaConf.to_container(cfg, resolve=True)
        run = wandb.init(
            project="merl_hrl",
            group=cfg.group,
            config=config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            save_code=True,  # optional
            name=cfg.tag if cfg.tag else None,
        )

    logger.info(f"Working directory: {os.getcwd()}")

    env_sim, env_real = make_sim_real_envs(cfg)
    nom_skills = get_nominal_skills(task=cfg.env.env_name)

    if cfg.comment:
        with open("README.txt", "w") as f:
            f.write(cfg.comment)

    if cfg.use_ray:
        ray.init(num_cpus=cfg.num_cpus + 2, num_gpus=cfg.num_gpus)

    # Discover failures
    if cfg.failures.learn:
        logger.info("Collecting failures in 'real' sim")
        logger.info("----------------------------------")
        cfg_fail = deepcopy(cfg)
        cfg_fail.env.reward_shaping = False
        cfg_fail.rl.reward.potential.enabled = False
        cfg_fail.rl.reward.eef_force.enabled = False
        # cfg_fail.rl.reward.action.enabled = True
        env = make_env(cfg_fail)
        fails, demos = collect_failures(env, nom_skills, cfg_fail)
    else:
        logger.info("Loading failures")
        logger.info("----------------")
        fails = pkl_load(
            join(to_absolute_path(cfg.data_dir), cfg.failures.fail_filename)
        )
        demos = pkl_load(
            join(to_absolute_path(cfg.data_dir), cfg.failures.demo_filename)
        )

    if cfg.failures.viz:
        for fail in fails:
            viz_states(env_sim, [fail])

    if cfg.failures.record:
        record_states(env_sim, fails)

    if cfg.failures.record_nominal_traj:
        record_nominal_traj(nom_skills, cfg.nevals, cfg)

    # Fit a value function using nominal skills
    if cfg.value_fn.learn:
        if "pos" in demos:
            all_demos = demos["pos"] + demos["neg"]
        else:
            all_demos = demos
        value_fn = fit_value_fn(all_demos, cfg.value_fn)
        cfg["value_fn"]["model_filename"] = "value_fn.pt"
        cfg["value_fn"]["scaler_filename"] = "value_fn_scaler.pkl"

    #  Cluster failures
    if cfg.clusters.learn:
        logger.info("Clustering failures")
        logger.info("-------------------")
        clusters = cluster_failures(fails, cfg)
    else:
        logger.info("Loading clusters")
        logger.info("----------------")
        clusters = pkl_load(join(to_absolute_path(
            cfg.data_dir), cfg.clusters.filename))

    # Recovery learning
    if cfg.recoveries.learn:
        logger.info("Learning recoveries")
        logger.info("-------------------")
        if cfg.recoveries.warmstart:
            recoveries = pkl_load(
                join(to_absolute_path(cfg.data_dir), cfg.recoveries.filename)
            )
        else:
            recoveries = None

        recoveries = learn_recoveries(
            clusters["train"], cfg, warmstart=recoveries)
    else:
        logger.info("Loading recoveries")
        logger.info("-------------------")
        # recoveries = pkl_load(
        # join(to_absolute_path(cfg.data_dir), cfg.recoveries.filename)
        # )

    # Evluate learned recoveries
    if cfg.evaluate.evaluate_recovery:
        eval_env = make_env(cfg, test=True)
        if cfg.evaluate.method == "recovery":
            eval_info = evaluate_recoveries(clusters["test"], eval_env, cfg)
            # eval_info = evaluate_recoveries(clusters["train"], eval_env, cfg)
        else:
            raise NotImplementedError

    elif cfg.evaluate.evaluate_chain:
        eval_env = make_env(cfg, test=True)
        if cfg.evaluate.method == "nominal":
            eval_info = evaluate_nominal_skills(
                eval_env, nom_skills, cfg.evaluate)
        else:
            subdirs = [
                f.path
                for f in os.scandir(
                    f"/home/aries/research/belief_srs/results/ral/{cfg.evaluate.env_name}/{cfg.evaluate.algo}/skills"
                )
                if f.is_dir()
            ]
            logger.info(f"Evaluating subdir policies: {subdirs}")
            infos = []
            for subdir in subdirs:
                logger.info(f"Evaluating subdir: {subdir}")
                cfg.evaluate.path_to_policy = join(
                    subdir, "policies/best_model.zip")
                cfg.evaluate.path_to_vnorm = join(
                    subdir, "policies/best_vecnorm.pkl")
                eval_info = evaluate_recoveries(
                    clusters["test"],
                    eval_env,
                    cfg,
                    eval_from_start=True,
                    nom_skills=None if cfg.evaluate.evaluate_policy else nom_skills,
                )
                recovery_succs = eval_info["recovery_succs"]
                infos.append(eval_info)
            succs = [info["succs"] for info in infos]
            nom_succs = [info["nom_succs"] for info in infos]
            logger.info("Nominal + recovery evaluation result:")
            logger.info("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
            logger.info(
                f"  Overall Mean SR: {np.mean(succs)*100}: {np.sum(succs)} / {np.concatenate(succs).size}"
            )
            logger.info(
                f"  Nominal mean SR: {np.mean(nom_succs)*100}: {np.sum(nom_succs)} / {np.concatenate(nom_succs).size}"
            )

    if cfg.use_wandb:
        run.finish()


if __name__ == "__main__":
    main()
