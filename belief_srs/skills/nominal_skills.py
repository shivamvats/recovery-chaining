from copy import deepcopy
import logging
import numpy as np
from robosuite.utils.transform_utils import *

from .robot_skill import RobotSkill

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)


class TargetSkill(RobotSkill):
    """Skill defined with respect to a target."""

    def __init__(self, target, offset=None, steps_per_action=30):
        super().__init__(steps_per_action)
        self.target = target
        if offset is None:
            offset = np.zeros(3)
        self.offset = offset


# Shelf skills
# ---------------

class SideGraspSkill(TargetSkill):
    skill_id = 0

    def _apply(self, env, obs, render=False):
        logger.debug("  grasp skill")
        total_rew, option_time = 0, 0
        discount, gamma = 1, 1.0 # 0.99
        hist = {"state": [], "reward": [], "skill": []}

        # true_state = env.observe_true_state()
        # dummy transition to reset option time
        # hist["state"].append(true_state)
        # hist["reward"].append(0)

        eef_pos, eef_mat = self.eef_pose(env)

        # # target_pos = obs[f"{self.target}_pos"]
        # target_pos = eef_pos

        # # pregrasp
        # eef_pos, eef_mat = self.eef_pose(env)
        # delta_rot1 = rotation_matrix(angle=np.pi / 2, direction=np.array([1, 0, 0]))[
        # :3, :3
        # ]
        # delta_rot2 = rotation_matrix(angle=np.pi / 2, direction=np.array([0, 1, 0]))[
        # :3, :3
        # ]
        # target_rot = np.matmul(delta_rot2, np.matmul(delta_rot1, eef_mat))
        # eef_ori = quat2axisangle(mat2quat(target_rot))
        # # target_pos[1] -= 0.1
        # action = np.concatenate([target_pos, eef_ori, [-1]])

        # obs, rew, done, info = self.apply_action(env, action, render)
        # total_rew += rew
        # #
        # hist["state"].extend(info["hist"]["state"])
        # hist["reward"].extend(info["hist"]["reward"])

        # if done:
        # return obs, total_rew, done, info

        # box_target_pos = obs[f"{self.target}_pos"]
        # target_pos = deepcopy(box_target_pos)
        # target_pos[1] -= 0.1
        # action = np.concatenate([target_pos, eef_ori, [-1]])
        # obs, rew, done, info = self.apply_action(env, action, render)
        # total_rew += rew
        # #
        # hist["state"].extend(info["hist"]["state"])
        # hist["reward"].extend(info["hist"]["reward"])

        # __import__('ipdb').set_trace()

        # if done:
        # return obs, total_rew, done, info

        # grasp pose
        box_dims = obs["box_dims"]
        eef_pos, eef_mat = self.eef_pose(env)
        eef_ori = quat2axisangle(mat2quat(eef_mat))
        target_pos = deepcopy(obs[f"{self.target}_pos"])
        target_pos[1] += 0.04 - box_dims[1]
        action = np.concatenate([target_pos, eef_ori, [-1]])

        obs, rew, done, info = self.apply_action(env, action, render)
        total_rew += rew
        #
        # not markovian
        # hist["state"].extend(info["hist"]["state"])
        # hist["reward"].extend(info["hist"]["reward"])
        # hist["state"].append(info["option"]["state"])
        # hist["reward"].append(info["option"]["reward"])

        hist["reward"].append(rew)
        hist["skill"].append(self.skill_id)
        hist["state"].append(env.observe_true_state(obs, mj_state=False))

        if done:
            info["hist"] = hist
            return obs, total_rew, done, info

        # grasp
        eef_pos, eef_mat = self.eef_pose(env)
        eef_ori = quat2axisangle(mat2quat(eef_mat))
        target_pos = eef_pos
        action = np.concatenate([target_pos, eef_ori, [1]])
        obs, rew, done, info = self.apply_action(env, action, render)
        total_rew += rew
        #
        # hist["state"].extend(info["hist"]["state"])
        # hist["reward"].extend(info["hist"]["reward"])
        # hist["state"].append(info["option"]["state"])
        # hist["reward"].append(info["option"]["reward"])

        hist["reward"].append(rew)
        hist["skill"].append(self.skill_id)
        hist["state"].append(env.observe_true_state(obs, mj_state=False))
        if done:
            # info["hist"] = dict(state=[info["option"]["state"]], reward=[total_rew])
            info["hist"] = hist
            return obs, total_rew, done, info

        # pickup
        # eef_pos, eef_mat = self.eef_pose(env)
        # eef_ori = quat2axisangle(mat2quat(eef_mat))
        # target_pos = eef_pos
        # # target_pos[2] += 0.1
        # target_pos[2] += 0.05
        # action = np.concatenate([target_pos, eef_ori, [1]])
        # obs, rew, done, info = self.apply_action(env, action, render)
        # total_rew += rew
        # #
        # hist["state"].extend(info["hist"]["state"])
        # hist["reward"].extend(info["hist"]["reward"])

        # info["hist"] = dict(state=[info["option"]["state"]], reward=[total_rew])
        info["hist"] = hist
        return obs, total_rew, done, info


class GotoShelfSkill(TargetSkill):
    """Could be replaced by a motion planner."""

    skill_id = 1

    def _apply(self, env, obs, render=False):
        logger.debug("  goto shelf skill")
        total_rew, option_time = 0, 0
        discount, gamma = 1, 1.0 # 0.99
        hist = {"state": [], "reward": [], "skill": []}

        # true_state = env.observe_true_state()
        # dummy transition to reset option time
        # hist["state"].append(true_state)
        # hist["reward"].append(0)

        eef_pos, eef_mat = self.eef_pose(env)
        eef_ori = quat2axisangle(mat2quat(eef_mat))

        target_pos = deepcopy(obs[f"{self.target}_pos"])

        # clear the base of the shelf
        target_pos[1] -= 0.15
        # target_pos[2] += 0.02
        action = np.concatenate([target_pos, eef_ori, [1]])

        obs, rew, done, info = self.apply_action(env, action, render)
        total_rew += rew

        hist["reward"].append(rew)
        hist["skill"].append(self.skill_id)
        hist["state"].append(env.observe_true_state(obs, mj_state=False))
        #
        # hist["state"].extend(info["hist"]["state"])
        # hist["reward"].extend(info["hist"]["reward"])
        # hist["state"].append(info["option"]["state"])
        # hist["reward"].append(info["option"]["reward"])

        if done:
            # info["hist"] = dict(state=[info["option"]["state"]], reward=[total_rew])
            info["hist"] = hist
            return obs, total_rew, done, info

        # eef_pos, eef_mat = self.eef_pose(env)
        # eef_ori = quat2axisangle(mat2quat(eef_mat))
        # target_pos = deepcopy(obs[f"{self.target}_pos"])
        # target_pos[2] = eef_pos[2]
        # action = np.concatenate([target_pos, eef_ori, [1]])

        # obs, rew, done, info = self.apply_action(env, action, render)
        # total_rew += rew
        # #
        # # hist["state"].extend(info["hist"]["state"])
        # # hist["reward"].extend(info["hist"]["reward"])
        # hist["state"].append(info["option"]["state"])
        # hist["reward"].append(info["option"]["reward"])

        # info["hist"] = hist
        # info["hist"] = dict(state=[info["option"]["state"]], reward=[total_rew])
        info["hist"] = hist
        return obs, total_rew, done, info


class ShelfPlaceSkill(TargetSkill):
    skill_id = 2

    def _apply(self, env, obs, render=False):
        logger.debug("  place skill")
        total_rew, option_time = 0, 0
        discount, gamma = 1, 1.0 # 0.99
        hist = {"state": [], "reward": [], "skill": []}

        # true_state = env.observe_true_state()
        # dummy transition to reset option time
        # hist["state"].append(true_state)
        # hist["reward"].append(0)

        eef_pos, eef_mat = self.eef_pose(env)
        eef_ori = quat2axisangle(mat2quat(eef_mat))

        # # release pose
        eef_pos, eef_mat = self.eef_pose(env)
        eef_ori = quat2axisangle(mat2quat(eef_mat))
        # target_pos = obs[f"{self.target}_pos"]
        target_pos = eef_pos
        eef_pos[1] += 0.15
        action = np.concatenate([target_pos, eef_ori, [1]])

        obs, rew, done, info = self.apply_action(env, action, render)
        total_rew += rew
        hist["reward"].append(rew)
        hist["skill"].append(self.skill_id)
        hist["state"].append(env.observe_true_state(obs, mj_state=False))
        #
        # hist["state"].extend(info["hist"]["state"])
        # hist["reward"].extend(info["hist"]["reward"])
        # hist["state"].append(info["option"]["state"])
        # hist["reward"].append(info["option"]["reward"])

        if done:
            # info["hist"] = dict(state=[info["option"]["state"]], reward=[total_rew])
            info["hist"] = hist
            return obs, total_rew, done, info

        # go down
        for _ in range(1):
            eef_pos, eef_mat = self.eef_pose(env)
            eef_ori = quat2axisangle(mat2quat(eef_mat))
            target_pos = eef_pos
            target_pos[2] -= 0.02
            action = np.concatenate([target_pos, eef_ori, [1]])
            obs, rew, done, info = self.apply_action(
                env, action, render, steps_per_action=10
            )
            total_rew += rew
            hist["reward"].append(rew)
            hist["skill"].append(self.skill_id)
            hist["state"].append(env.observe_true_state(obs, mj_state=False))
            collision = obs["collision"]
            if collision or done:
                break
            # hist["state"].append(info["option"]["state"])
            # hist["reward"].append(info["option"]["reward"])

        if done:
            # info["hist"] = dict(state=[info["option"]["state"]], reward=[total_rew])
            info["hist"] = hist
            return obs, total_rew, done, info

        eef_pos, eef_mat = self.eef_pose(env)
        eef_ori = quat2axisangle(mat2quat(eef_mat))
        gripper_pos = obs["robot0_gripper_qpos"]
        gripper_closed = gripper_pos[0] - gripper_pos[1] <= 0.06

        if gripper_closed:
            # release
            target_pos = eef_pos
            action = np.concatenate([target_pos, eef_ori, [-1]])
            obs, rew, done, info = self.apply_action(env, action, render)
            total_rew += rew
            hist["reward"].append(rew)
            hist["skill"].append(self.skill_id)
            hist["state"].append(env.observe_true_state(obs, mj_state=False))
            #
            # hist["state"].extend(info["hist"]["state"])
            # hist["reward"].extend(info["hist"]["reward"])
            # hist["state"].append(info["option"]["state"])
            # hist["reward"].append(info["option"]["reward"])

            if done:
                # info["hist"] = dict(state=[info["option"]["state"]], reward=[total_rew])
                info["hist"] = hist
                return obs, total_rew, done, info

        # retract
        eef_pos, eef_mat = self.eef_pose(env)
        eef_ori = quat2axisangle(mat2quat(eef_mat))
        target_pos = eef_pos
        target_pos[1] -= 0.2
        action = np.concatenate([target_pos, eef_ori, [-1]])
        obs, rew, done, info = self.apply_action(env, action, render)
        total_rew += rew
        hist["reward"].append(rew)
        hist["skill"].append(self.skill_id)
        hist["state"].append(env.observe_true_state(obs, mj_state=False))
        # hist["state"].extend(info["hist"]["state"])
        # hist["reward"].extend(info["hist"]["reward"])
        # hist["state"].append(info["option"]["state"])
        # hist["reward"].append(info["option"]["reward"])

        # info["hist"] = hist
        # info["hist"] = dict(state=[info["option"]["state"]], reward=[total_rew])
        info["hist"] = hist
        logger.debug("  end of place skill")
        return obs, total_rew, done, info


# Pick-place skills
# ----------------------

class GotoGraspSkill(TargetSkill):
    skill_id = 0

    # Markovian option
    def _apply(self, env, obs, render=False):
        total_rew = 0

        #---------
        # TODO figure out how to discount correctly!
        # ------
        discount, gamma = 1, 1.0 # 0.99
        hist = {"state": [], "reward": [], "skill": [], "action": []}

        eef_pos, eef_mat = self.eef_pose(env)
        eef_ori = quat2axisangle(mat2quat(eef_mat))

        target_pos = obs[f"{self.target}_pos"]

        # pregrasp
        eef_pos, eef_mat = self.eef_pose(env)
        eef_ori = quat2axisangle(mat2quat(eef_mat))
        top_offset = env.objects[env.object_to_id[self.target.lower()]].top_offset
        # target_pos[2] += (top_offset[2] / 2 - 0.01)
        target_pos[2] += (top_offset[2] / 2 + 0.05)
        action = np.concatenate([target_pos, eef_ori, [-1]])

        for _ in range(15):
            obs, rew, done, info = env.step(action)
            total_rew += discount*rew
            discount *= gamma
            if render:
                env.render()
            hist["state"].append(env.observe_true_state(obs, mj_state=False))
            hist["reward"].append(rew)
            hist["skill"].append(self.skill_id)
            hist["action"].append(action)
            if done or np.linalg.norm(obs["robot0_eef_pos"] - target_pos) < 0.005:
                break

        info["hist"] = hist
        info['discount'] = discount

        return obs, total_rew, done, info


class PickupSkill(TargetSkill):
    skill_id = 1

    # Markovian option
    def _apply(self, env, obs, render=False):
        total_rew = 0
        discount, gamma = 1, 1 #0.99
        hist = {"state": [], "reward": [], "skill": [], "action": []}

        for _ in range(15):
            if env._check_grasp(
                gripper=env.robots[0].gripper,
                object_geoms=env.objects[env.object_to_id[self.target.lower()]].contact_geoms,
            ):
                # lift
                logger.debug("  Lifting")
                target_pos = obs[f"{self.target}_pos"]
                logger.debug(f"    Target pos: {target_pos}")
                eef_pos, eef_mat = self.eef_pose(env)
                eef_ori = quat2axisangle(mat2quat(eef_mat))
                target_pos[2] = env.bin1_pos[2] + 0.2
                action = np.concatenate([target_pos, eef_ori, [1]])
                dist_to_target = np.linalg.norm(eef_pos - target_pos)
                obs, rew, done, info = env.step(action)
                if render:
                    env.render()
 
                total_rew += discount*rew
                discount *= gamma

                hist["state"].append(env.observe_true_state(obs, mj_state=False))
                hist["reward"].append(rew)
                hist["skill"].append(self.skill_id)
                hist["action"].append(action)
                if done or dist_to_target < 0.005:
                    break

            else:
                eef_pos, eef_mat = self.eef_pose(env)
                eef_ori = quat2axisangle(mat2quat(eef_mat))
                top_offset = env.objects[env.object_to_id[self.target.lower()]].top_offset
                target_pos = obs[f"{self.target}_pos"]
                target_pos[2] += (top_offset[2] / 2 - 0.01)

                if np.linalg.norm(eef_pos - target_pos) > 0.005:
                    logger.debug("  going to grasp")
                    action = np.concatenate([target_pos, eef_ori, [-1]])
                    obs, rew, done, info = env.step(action)
                    total_rew += discount*rew
                    discount *= gamma
                    if render:
                        env.render()
                    hist["state"].append(env.observe_true_state(obs, mj_state=False))
                    hist["reward"].append(rew)
                    hist["skill"].append(self.skill_id)
                    hist["action"].append(action)
                    if done:
                        break

                else:
                    # close gripper
                    logger.debug("  closing gripper")
                    eef_pos, eef_mat = self.eef_pose(env)
                    eef_ori = quat2axisangle(mat2quat(eef_mat))
                    target_pos = eef_pos
                    action = np.concatenate([target_pos, eef_ori, [1]])
                    obs, rew, done, info = self.apply_action(env, action, render)
                    total_rew += discount*rew
                    discount *= gamma
                    hist["state"].append(env.observe_true_state(obs, mj_state=False))
                    hist["reward"].append(rew)
                    hist["skill"].append(self.skill_id)
                    hist["action"].append(action)

                    if render:
                        env.render()
                    if done:
                        break

        info["hist"] = hist
        info['discount'] = discount

        return obs, total_rew, done, info


class GotoGoalSkill(TargetSkill):
    skill_id = 2

    def _apply(self, env, obs, render=False):
        hist = {"state": [], "reward": [], "skill": [], "action": []}
        total_rew = 0
        discount, gamma = 1, 1 #0.99

        bin_id = env.object_to_id[self.target.lower()]
        bin_x_low = env.bin2_pos[0]
        bin_y_low = env.bin2_pos[1]
        if bin_id == 0 or bin_id == 2:
            bin_x_low -= env.bin_size[0] / 2
        if bin_id < 2:
            bin_y_low -= env.bin_size[1] / 2

        bin_x_high = bin_x_low + env.bin_size[0] / 2
        bin_y_high = bin_y_low + env.bin_size[1] / 2
        target_pos = np.array(
            [
                (bin_x_low + bin_x_high) / 2,
                (bin_y_low + bin_y_high) / 2,
                env.bin2_pos[2] + 0.2,
            ]
        )

        for t in range(15):
            eef_pos, eef_mat = self.eef_pose(env)
            eef_ori = quat2axisangle(mat2quat(eef_mat))
            # goto target_pos
            if np.linalg.norm(eef_pos - target_pos) > 0.05:
                target = (target_pos + eef_pos) / 2
            else:
                target = target_pos
            action = np.concatenate([target, eef_ori, [1]])
            obs, rew, done, info = env.step(action)
            total_rew += discount*rew
            discount *= discount
            if render:
                env.render()
            hist["state"].append(env.observe_true_state(obs, mj_state=False))
            hist["reward"].append(rew)
            hist["skill"].append(self.skill_id)
            hist["action"].append(action)
            if done or np.linalg.norm(eef_pos - target_pos) < 0.005:
                break

        info["hist"] = hist
        info['discount'] = discount

        logger.debug(f"  goto reward: {total_rew}")
        return obs, total_rew, done, info


class PlaceSkill(TargetSkill):
    skill_id = 3

    def _apply(self, env, obs, render=False):
        hist = {"state": [], "reward": [], "skill": [], "action": []}
        total_rew = 0
        discount, gamma = 1, 1 #0.99

        bin_id = env.object_to_id[self.target.lower()]
        bin_x_low = env.bin2_pos[0]
        bin_y_low = env.bin2_pos[1]
        if bin_id == 0 or bin_id == 2:
            bin_x_low -= env.bin_size[0] / 2
        if bin_id < 2:
            bin_y_low -= env.bin_size[1] / 2

        bin_x_high = bin_x_low + env.bin_size[0] / 2
        bin_y_high = bin_y_low + env.bin_size[1] / 2
        target_pos = np.array(
            [
                (bin_x_low + bin_x_high) / 2,
                (bin_y_low + bin_y_high) / 2,
                env.bin2_pos[2] + 0.1,
            ]
        )

        for t in range(10):
            eef_pos, eef_mat = self.eef_pose(env)
            eef_ori = quat2axisangle(mat2quat(eef_mat))
            if np.linalg.norm(eef_pos - target_pos) < 0.01:
                # open_gripper
                target_pos = eef_pos
                action = np.concatenate([target_pos, eef_ori, [-1]])
                obs, rew, done, info = self.apply_action(env, action, render,
                                                         steps_per_action=10)
                logger.debug(f"  opening rew: {rew}")
                total_rew += discount*rew
                discount *= gamma
                hist["state"].append(env.observe_true_state(obs, mj_state=False))
                hist["reward"].append(rew)
                hist["skill"].append(self.skill_id)
                hist["action"].append(action)
                break
            else:
                # goto target_pos
                action = np.concatenate([target_pos, eef_ori, [1]])
                obs, rew, done, info = env.step(action)
                total_rew += discount*rew
                discount *= gamma
                if render:
                    env.render()
                hist["state"].append(env.observe_true_state(obs, mj_state=False))
                hist["reward"].append(rew)
                hist["skill"].append(self.skill_id)
                hist["action"].append(action)
                if done:
                    break

        info["hist"] = hist
        info['discount'] = discount

        logger.debug(f"  place reward: {total_rew}")
        return obs, total_rew, done, info


class SinglePickupSkill(TargetSkill):
    skill_id = 0

    # Non-markovian option
    def _apply(self, env, obs, render=False):
        total_rew = 0
        discount, gamma = 1, 1 #0.99
        hist = {"state": [], "reward": [], "skill": []}

        target_pos = obs[f"{self.target}_pos"]
        eef_pos, eef_mat = self.eef_pose(env)
        eef_ori = quat2axisangle(mat2quat(eef_mat))
        top_offset = env.objects[env.object_to_id[self.target.lower()]].top_offset
        target_pos[2] += (top_offset[2] / 2 + 0.05)
        action = np.concatenate([target_pos, eef_ori, [-1]])

        for _ in range(15):
            obs, rew, done, info = env.step(action)
            total_rew += discount*rew
            discount *= gamma
            if render:
                env.render()
            hist["state"].append(env.observe_true_state(obs, mj_state=False))
            hist["reward"].append(rew)
            hist["skill"].append(self.skill_id)
            if done or np.linalg.norm(obs["robot0_eef_pos"] - target_pos) < 0.005:
                break

        for _ in range(15):
            if env._check_grasp(
                gripper=env.robots[0].gripper,
                object_geoms=env.objects[env.object_to_id[self.target.lower()]].contact_geoms,
            ):
                # lift
                logger.debug("  Lifting")
                target_pos = obs[f"{self.target}_pos"]
                logger.debug(f"    Target pos: {target_pos}")
                eef_pos, eef_mat = self.eef_pose(env)
                eef_ori = quat2axisangle(mat2quat(eef_mat))
                target_pos[2] = env.bin1_pos[2] + 0.2
                action = np.concatenate([target_pos, eef_ori, [1]])
                dist_to_target = np.linalg.norm(eef_pos - target_pos)
                obs, rew, done, info = env.step(action)
                if render:
                    env.render()
 
                total_rew += discount*rew
                discount *= gamma

                hist["state"].append(env.observe_true_state(obs, mj_state=False))
                hist["reward"].append(rew)
                hist["skill"].append(self.skill_id)
                if done or dist_to_target < 0.005:
                    break

            else:
                eef_pos, eef_mat = self.eef_pose(env)
                eef_ori = quat2axisangle(mat2quat(eef_mat))
                top_offset = env.objects[env.object_to_id[self.target.lower()]].top_offset
                target_pos = obs[f"{self.target}_pos"]
                target_pos[2] += (top_offset[2] / 2 - 0.01)

                if np.linalg.norm(eef_pos - target_pos) > 0.005:
                    logger.debug("  going to grasp")
                    action = np.concatenate([target_pos, eef_ori, [-1]])
                    obs, rew, done, info = env.step(action)
                    total_rew += discount*rew
                    discount *= gamma
                    if render:
                        env.render()
                    hist["state"].append(env.observe_true_state(obs, mj_state=False))
                    hist["reward"].append(rew)
                    hist["skill"].append(self.skill_id)
                    if done:
                        break

                else:
                    # close gripper
                    logger.debug("  closing gripper")
                    eef_pos, eef_mat = self.eef_pose(env)
                    eef_ori = quat2axisangle(mat2quat(eef_mat))
                    target_pos = eef_pos
                    action = np.concatenate([target_pos, eef_ori, [1]])
                    obs, rew, done, info = self.apply_action(env, action, render)
                    total_rew += discount*rew
                    discount *= gamma
                    hist["state"].append(env.observe_true_state(obs, mj_state=False))
                    hist["reward"].append(rew)
                    hist["skill"].append(self.skill_id)

                    if render:
                        env.render()
                    if done:
                        break

        info["hist"] = hist
        info['discount'] = discount

        return obs, total_rew, done, info

#---------------------

class BlockPlaceSkill(TargetSkill):
    def _apply(self, env, obs, render=False):
        hist = {"state": [], "reward": []}
        eef_pos, eef_mat = self.eef_pose(env)
        eef_ori = quat2axisangle(mat2quat(eef_mat))

        target_pos = obs[f"{self.target}_pos"] + self.offset

        # pre-release
        eef_pos, eef_mat = self.eef_pose(env)
        eef_ori = quat2axisangle(mat2quat(eef_mat))
        target_pos[2] += 0.08
        action = np.concatenate([target_pos, eef_ori, [1]])

        obs, rew, done, info = self.apply_action(env, action, render)
        #
        hist["state"].extend(info["hist"]["state"])
        hist["reward"].extend(info["hist"]["reward"])

        # release pose
        eef_pos, eef_mat = self.eef_pose(env)
        eef_ori = quat2axisangle(mat2quat(eef_mat))
        target_pos = obs[f"{self.target}_pos"] + self.offset
        target_pos[2] += 0.04
        action = np.concatenate([target_pos, eef_ori, [1]])

        obs, rew, done, info = self.apply_action(env, action, render)
        #
        hist["state"].extend(info["hist"]["state"])
        hist["reward"].extend(info["hist"]["reward"])

        # release
        eef_pos, eef_mat = self.eef_pose(env)
        eef_ori = quat2axisangle(mat2quat(eef_mat))
        target_pos = eef_pos
        action = np.concatenate([target_pos, eef_ori, [-1]])
        obs, rew, done, info = self.apply_action(env, action, render)
        #
        hist["state"].extend(info["hist"]["state"])
        hist["reward"].extend(info["hist"]["reward"])

        # retract
        eef_pos, eef_mat = self.eef_pose(env)
        eef_ori = quat2axisangle(mat2quat(eef_mat))
        target_pos = eef_pos
        target_pos[2] += 0.1
        action = np.concatenate([target_pos, eef_ori, [-1]])
        obs, rew, done, info = self.apply_action(env, action, render)
        #
        hist["state"].extend(info["hist"]["state"])
        hist["reward"].extend(info["hist"]["reward"])

        info["hist"] = hist

        rew = np.sum(info["hist"]["reward"])

        return obs, rew, done, info


class TablePlaceSkill(TargetSkill):
    def _apply(self, env, obs, render=False):
        hist = {"state": [], "reward": []}
        eef_pos, eef_mat = self.eef_pose(env)
        eef_ori = quat2axisangle(mat2quat(eef_mat))

        target_pos = obs[f"{self.target}_pos"] + self.offset

        # pre-release
        eef_pos, eef_mat = self.eef_pose(env)
        eef_ori = quat2axisangle(mat2quat(eef_mat))
        target_pos[2] += 0.04
        action = np.concatenate([target_pos, eef_ori, [1]])

        obs, rew, done, info = self.apply_action(env, action, render)
        #
        hist["state"].extend(info["hist"]["state"])
        hist["reward"].extend(info["hist"]["reward"])

        if done:
            return obs, rew, done, info

        # release pose
        eef_pos, eef_mat = self.eef_pose(env)
        eef_ori = quat2axisangle(mat2quat(eef_mat))
        target_pos = obs[f"{self.target}_pos"] + self.offset
        # target_pos[1] += -0.01
        # target_pos[2] += 0.04
        action = np.concatenate([target_pos, eef_ori, [1]])

        obs, rew, done, info = self.apply_action(env, action, render)
        #
        hist["state"].extend(info["hist"]["state"])
        hist["reward"].extend(info["hist"]["reward"])

        if done:
            return obs, rew, done, info

        # release
        eef_pos, eef_mat = self.eef_pose(env)
        eef_ori = quat2axisangle(mat2quat(eef_mat))
        target_pos = eef_pos
        action = np.concatenate([target_pos, eef_ori, [-1]])
        obs, rew, done, info = self.apply_action(env, action, render)
        #
        hist["state"].extend(info["hist"]["state"])
        hist["reward"].extend(info["hist"]["reward"])

        if done:
            return obs, rew, done, info

        # retract
        eef_pos, eef_mat = self.eef_pose(env)
        eef_ori = quat2axisangle(mat2quat(eef_mat))
        target_pos = eef_pos
        target_pos[2] += 0.1
        action = np.concatenate([target_pos, eef_ori, [-1]])
        obs, rew, done, info = self.apply_action(env, action, render)
        #
        hist["state"].extend(info["hist"]["state"])
        hist["reward"].extend(info["hist"]["reward"])

        info["hist"] = hist

        rew = np.sum(info["hist"]["reward"])
        return obs, rew, done, info


# Real robot skills
# -----------------
