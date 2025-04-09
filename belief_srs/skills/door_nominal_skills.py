"""
Open-loop controllers
"""

from collections.abc import Mapping
import numpy as np

from autolab_core import RigidTransform
from .robot_skill import RobotSkill
from belief_srs.utils.transforms import *


class ReachAndGraspHandleSkill(RobotSkill):
    """Moves to the pre-grasp pose for grasping the handle."""
    skill_id = 0
    def __init__(self, preconds=None, **kwargs):
        super().__init__(**kwargs)

        self.preconds = preconds

    def _apply(self, env, obs, context=None, render=False, interpolate=False):

        time_per_action = self.time_per_action
        hist = {"state": [], "reward": [], "skill": []}

        TIMESTEPS_PER_ACTION = int(time_per_action / env.control_timestep)

        if isinstance(obs, Mapping):
            obs_dict = obs
        else:
            obs_dict = env.unflatten_obs(obs)

        # middle point of the handle
        ee_pos, ee_mat = self.eef_pose(env)
        ee_ori = quat2axisangle(mat2quat(ee_mat))
        handle_pos = obs_dict['handle_pos']

        actions = []

        # 90deg about x axis
        rot_x = mat_about_x(np.pi/2)
        # rot_x = mat_about_x(0)

        # rot_z = mat_about_z(door_anti_normal)
        rot_z = mat_about_z(-np.pi/2)
        target_rot1 = np.matmul(rot_x, rot_z)
        # target_rot1 = rot_x
        target_rot1 = quat2axisangle(mat2quat(target_rot1))

        # before pre-grasp
        grasp_pos1 = ee_pos
        action1 = np.concatenate([grasp_pos1, target_rot1, [-1.0]])
        actions.append(action1)

        offset2 = np.array([0.0, 0.0, 0.0])
        grasp_pos2 = handle_pos + offset2
        grasp_ori2 = ee_ori
        action2 = np.concatenate([grasp_pos2, target_rot1, [-1.0]])
        actions.append(action2)

        offset2 = np.array([0.0, -0.06, 0.0])
        grasp_pos2 = handle_pos + offset2
        grasp_ori2 = ee_ori
        action3 = np.concatenate([grasp_pos2, target_rot1, [1.0]])
        actions.append(action3)

        for action in actions:
            obs, rew, done, info = self.apply_action(env, action, render)
            hist["state"].extend(info["hist"]["state"])
            hist["reward"].extend(info["hist"]["reward"])
            hist["skill"].append(self.skill_id)

        # __import__('ipdb').set_trace()
        # if isinstance(obs, Mapping):
            # obs_dict = obs
        # else:
            # obs_dict = env.unflatten_obs(obs)

        # ee_pos = obs_dict['robot0_eef_pos']
        # ee_quat = obs_dict['robot0_eef_quat']
        # ee_axisangle = T.quat2axisangle(ee_quat)

        # action = np.concatenate([ee_pos, ee_axisangle, [1.0]])
        # for _ in range(TIMESTEPS_PER_ACTION):
            # obs, rew, done, info = env.step(action)
            # if render:
                # env.render()

        info["hist"] = hist
        rew = np.sum(info["hist"]["reward"])

        return obs, rew, done, info


class RotateHandleSkill(RobotSkill):
    """Rotates the handle so as to open it"""

    def __init__(self, preconds=None, **kwargs):
        super().__init__(**kwargs)

        self.preconds = preconds

    def precondition_satisfied(self, state, context=None, env=None):
        if env:
            precond = env.check_handle_grasped(state)
        else:
            precond = 1.0

        if self.preconds:
            return precond and self.preconds.is_satisfied(state, context)
        else:
            return precond

    def termcondition_satisfied(self, state, context=None):
        return False

    def apply(self, env, obs, context=None, render=False, interpolate=False):

        time_per_action = self.time_per_action

        TIMESTEPS_PER_ACTION = int(time_per_action / env.control_timestep)

        if isinstance(obs, Mapping):
            obs_dict = obs
        else:
            obs_dict = env.unflatten_obs(obs)

        # middle point of the handle
        ee_pos = obs_dict['robot0_eef_pos']
        ee_quat = obs_dict['robot0_eef_quat']
        ee_mat = T.quat2mat(ee_quat)
        handle_pos = obs_dict['handle_pos']
        # door_theta = obs_dict['door:pose/theta']
        # handle_dims = obs_dict['handle:dims']
        handle_length = 0.08 #handle_dims[0]
        # handle_cor_pos = obs_dict['handle_center_of_rotation:pose/position']

        actions = []

        current_ee_T = RigidTransform(translation=ee_pos,
                              rotation=T.quat2mat(ee_quat))

        target_pos = np.array([handle_cor_pos[0], # + 0.02,
                               handle_cor_pos[1],
                               handle_cor_pos[2] - handle_length
                               ])
        rot_mat = mat_about_y(-np.pi/2 - np.pi/12)
        target_mat = np.matmul(rot_mat, ee_mat)
        # target_rot = T.quat2axisangle(T.mat2quat(target_mat))

        start_T = RigidTransform(translation=ee_pos,
                               rotation=ee_mat)
        target_T = RigidTransform(translation=target_pos,
                                  rotation=target_mat)
        interps_T = start_T.linear_trajectory_to(target_T, 3)

        for interp in interps_T[1:]:
            action = np.concatenate([interp.translation,
                                     interp.axis_angle,
                                     [1.0]])
            # rot_y = mat_about_y(- handle_theta)
            actions.append(action)

        for action in actions:
            for _ in range(TIMESTEPS_PER_ACTION):
                obs, rew, done, info = env.step(action)
                if render:
                    env.render()

            # align with handle
            # obs = env.unflatten_obs(obs)
            # ee_pos = obs['robot_eef:pose/position']
            # ee_quat = obs['robot_eef:pose/quat']
            # handle_theta = obs['handle:pose/theta'][0]
            # # align with handle about y axis
            # rot_mat = mat_about_y(-handle_theta)
            # target_mat = np.matmul(rot_mat, ee_mat)
            # target_rot = T.quat2axisangle(T.mat2quat(target_mat))
            # target_pos = ee_pos

            # align_action = np.concatenate([target_pos,
                                            # target_rot,
                                            # [0]])
            # for _ in range(100):
                # obs, rew, done, info = env.step(align_action)
                # if render:
                    # env.render()

        return obs, rew, done, info


class PullHandleSkill(RobotSkill):
    """Pulls the handle to open the door"""

    skill_id = 1

    def __init__(self, preconds=None, **kwargs):
        super().__init__(**kwargs)

        self.preconds = preconds

    def _apply(self, env, obs, context=None, render=False, interpolate=False):

        time_per_action = self.time_per_action
        hist = {"state": [], "reward": [], "skill": []}

        TIMESTEPS_PER_ACTION = int(time_per_action / env.control_timestep)

        if isinstance(obs, Mapping):
            obs_dict = obs
        else:
            obs_dict = env.unflatten_obs(obs)

        # middle point of the handle
        ee_pos, ee_mat = self.eef_pose(env)
        ee_ori = quat2axisangle(mat2quat(ee_mat))
        handle_length = 0.08 #handle_dims[0]
        door_theta = obs_dict['hinge_qpos']

        actions = []

        target_pos = ee_pos + np.array([0, 0.2, 0])
        action1 = np.concatenate([target_pos, ee_ori, [1.0]])
        actions.append(action1)

        for action in actions:
            obs, rew, done, info = self.apply_action(env, action, render)
            hist["state"].extend(info["hist"]["state"])
            hist["reward"].extend(info["hist"]["reward"])
            hist["skill"].append(self.skill_id)
        info["hist"] = hist
        rew = np.sum(info["hist"]["reward"])

        return obs, rew, done, info
