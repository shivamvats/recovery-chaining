"""
Moves the shelf to the right of the robot.
"""

import collections
from copy import copy, deepcopy
import logging
import numpy as np
import os
import pyquaternion as pq
from time import sleep

from autolab_core import RigidTransform as RT
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.utils.observables import create_gaussian_noise_corrupter
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import (
    SequentialCompositeSampler,
    UniformRandomSampler,
)
from robosuite.utils.observables import *
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import (
    array_to_string,
    find_elements,
    xml_path_completion,
)
from robosuite.utils.mjcf_utils import array_to_string
import robosuite.utils.transform_utils as TU
from robosuite.models.objects import (
    BreadObject,
    CanObject,
    CerealObject,
    MilkObject,
)

import belief_srs
from .shelf_env import ShelfEnv
import belief_srs.utils.transforms as T
from belief_srs.utils.utils import *
from belief_srs.models.objects.shelf import Shelf
from belief_srs.models.grippers.schunk_gripper import SchunkGripper

logger = logging.getLogger(__name__)
# logger.setLevel("DEBUG")
logger.setLevel("INFO")


class ShelfEnvReal(ShelfEnv):
    """
    The task is to place a block at the edge of a table under uncertain block
    and edge position estimation.

    The task is irrecoverable if the cube is dropped from the table.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        self._load_robots()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](
            self.table_full_size[0]
        )
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="bluewood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        boxmat = CustomMaterial(
            texture="WoodLight",
            tex_name="woodlight",
            mat_name="woodlight_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        self.box = BoxObject(
            name="box",
            # cheezit box
            # size_min=[0.03, 0.079, 0.106],
            # size_max=[0.03, 0.079, 0.106],
            # size_min=[0.02, 0.06, 0.08],
            # size_max=[0.02, 0.06, 0.08],
            size_min=[0.02, 0.06, 0.08],
            size_max=[0.02, 0.08, 0.09],
            rgba=[1, 0, 0, 1],
            material=boxmat,
            # margin=1,
            # gap=1,
            # density=100,
            density=np.random.uniform(
                self.task_dist["box_density"][0], self.task_dist["box_density"][1]
            ),
        )

        self.clutters = []
        if self.task_dist["clutter"]["enable"]:
            # clutter_objs = [CanObject, CerealObject, MilkObject, CanObject, CerealObject]
            for i in range(self.task_dist["clutter"]["n_clutter"]):
                # clutter = clutter_objs[i](name=f"clutter{i}")
                clutter = BoxObject(
                    name=f"clutter{i}",
                    size_min=[0.02, 0.06, 0.08],
                    size_max=[0.02, 0.08, 0.09],
                    rgba=[1, 0, 0, 1],
                    material=bluewood,
                    density=100,
                )
                self.clutters.append(clutter)

        # Add sideview
        mujoco_arena.set_camera(
            camera_name="sideview",
            pos=[-0.05651774593317116, -1.2761224129427358, 1.4879572214102434],
            quat=[-0.8064181, -0.5912228, 0.00687796, 0.00990507],
        )

        self.shelf = Shelf(
            "shelf",
            # bin_size=(0.3, 0.3, 0.2), # original tall
            # bin_size=(0.3, 0.25, 0.2),  # shorter
            # bin_size=(0.3, 0.21, 0.2),  # shorter
            bin_size=(
                0.3,
                np.random.uniform(
                    self.task_dist["shelf_height"][0], self.task_dist["shelf_height"][1]
                ),
                0.2,
            ),
            # transparent_walls=False)
            transparent_walls=True,
        )

        # Create placement initializer
        self._get_placement_initializer()

        mj_objects = [self.shelf, self.box]
        if self.task_dist["clutter"]["enable"]:
            for clutter in self.clutters:
                mj_objects.append(clutter)

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=mj_objects,
        )

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        # Get robot prefix and define observables modality
        pf = self.robots[0].robot_model.naming_prefix
        modality = "object"

        # shelf
        @sensor(modality=modality)
        def target_pos(obs_cache):
            # TODO randomly sample goal on the shelf
            target_pos = deepcopy(self._target_pos)
            # far corner
            shelf_dims = self.shelf.dims
            # target_pos[0] += shelf_dims[0]/2 - 0.025
            target_pos[1] += 0.05
            return target_pos

        @sensor(modality=modality)
        def box_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.box_body_id])

        @sensor(modality=modality)
        def box_dims(obs_cache):
            return np.array([0.015, 0.07, 0.85])

        @sensor(modality=modality)
        def shelf_height(obs_cache):
            return self.shelf.dims[2]

        @sensor(modality=modality)
        def slip(obs_cache):
            # slip w.r.t when the robot grasped the object
            if self._box_quat_start is not None:
                curr_box_quat = self.sim.data.body_xquat[self.box_body_id]
                q1 = pq.Quaternion(self._box_quat_start)
                q2 = pq.Quaternion(curr_box_quat)
                rot_distance = pq.Quaternion.distance(q1, q2)
                slip = rot_distance
                # logger.info(f"  start: {q1}")
                # logger.info(f"  curr: {q2}")

            else:
                slip = -1
            # if slip != -1:
            # logger.info(f"    Slip: {slip}")
            return np.array([slip])

        @sensor(modality=modality)
        def obj_contacts(obs_cache):
            """
            specifies the cardinal direction in which the object in the eef or
            the eef collided.

            can be computed based on eef F/T signal.
            """
            include_geoms = [
                "shelf_base",
                "shelf_wall0",
                "shelf_wall1",
                "shelf_wall2",
                "shelf_wall3",
                "table_collision",
            ]
            include_geom_ids = [
                self.sim.model.geom_name2id(geom) for geom in include_geoms
            ]
            box_geom_id = self.sim.model.geom_name2id("box_g0")

            contacts = np.zeros(6)  # one-hot of 6 orthogonal directions
            for con in self.sim.data.contact:
                geom1, geom2 = con.geom1, con.geom2
                if box_geom_id == geom1 and geom2 in include_geom_ids:
                    con_normal = con.frame[:3]
                elif geom1 in include_geom_ids and box_geom_id == geom2:
                    con_normal = con.frame[:3]
                else:
                    con_normal = None

                if con_normal is not None:
                    major_axis = np.argmax(np.abs(con_normal))
                    sign = 0 if con_normal[major_axis] > 0 else 1
                    index = major_axis * 2 + sign
                    contacts[index] = 1
                    logger.debug(
                        f"  ({self.sim.model.geom_id2name(geom1)}, {self.sim.model.geom_id2name(geom2)}), contacts: {contacts}"
                    )
                    logger.debug(f"    pos: {con.pos}")
                else:
                    logger.debug("  No contact")

            return contacts

        @sensor(modality=modality)
        def collision(obs_cache):
            """
            DEPRECATED
            """
            eef_torques = self.robots[0].ee_torque
            return np.array([self._is_collision(), np.sign(eef_torques[0])], dtype=int)

        @sensor(modality=modality)
        def table_point(obs_cache):
            return deepcopy(self.table_offset)

        sensors = [
            target_pos,
            box_pos,
            slip,
            table_point,
            box_dims,
            shelf_height,
            collision,
            obj_contacts,
        ]
        names = [s.__name__ for s in sensors]

        # Create observables
        for name, s in zip(names, sensors):
            if name == "target_pos":
                corrupter = create_gaussian_noise_corrupter(
                    mean=np.zeros(3),
                    std=np.array([0.01, 0.01, 0.02]),
                )
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    corrupter=corrupter,
                    # corrupter=None,
                    sampling_rate=self.control_freq,
                )
            elif name == "box_pos":
                corrupter = create_gaussian_noise_corrupter(
                    mean=np.zeros(3),
                    std=np.array([0.01, 0.01, 0.02]),
                )
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    corrupter=corrupter,
                    # corrupter=None,
                    sampling_rate=self.control_freq,
                )
            else:
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    corrupter=None,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        self.stage = 0

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:
            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_name, (obj_pos, obj_quat, obj) in object_placements.items():
                if obj_name != "shelf":
                    self.sim.data.set_joint_qpos(
                        obj.joints[0],
                        np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                    )
            shelf_pos, shelf_quat, _ = object_placements[self.shelf.name]
            shelf_body_id = self.sim.model.body_name2id(self.shelf.root_body)
            self.sim.model.body_pos[shelf_body_id] = shelf_pos
            self.sim.model.body_quat[shelf_body_id] = shelf_quat

            # choose new target pos
            target_pos = np.array(shelf_pos)
            target_pos[2] += self.box.top_offset[2] - self.shelf.dims[2] / 2
            self._target_pos = target_pos

        self._box_quat_start = None

    def _check_success(self):
        """
        Check if box has been placed at the target pos.

        """
        gripper_geoms = [
            self.robots[0].gripper.important_geoms["left_finger"],
            self.robots[0].gripper.important_geoms["right_finger"],
        ]

        target_pos = self._target_pos
        box_pos = self.sim.data.body_xpos[self.box_body_id]
        box_quat = self.sim.data.body_xquat[self.box_body_id]

        robot_touching_box = any(
            [self.check_contact(geom, self.box) for geom in gripper_geoms]
        )
        box_on_shelf = self.check_contact(self.box, self.shelf)
        displace_bw_box_and_target = np.abs(target_pos - box_pos)
        # dist_bw_box_and_target = np.linalg.norm(target_pos - box_pos)
        q1 = pq.Quaternion(box_quat)
        q2 = pq.Quaternion(matrix=np.eye(3))
        rot_distance = pq.Quaternion.distance(q1, q2)

        if box_on_shelf and not robot_touching_box:
            # if dist_bw_box_and_target < 0.05 and rot_distance < 0.1:
            if rot_distance < 0.1 and (
                displace_bw_box_and_target[0] < 0.05
                and displace_bw_box_and_target[1] < 0.05
                and displace_bw_box_and_target[2] < 0.05
            ):
                return True

        return False

    def _check_terminated(self):
        """
        Terminate if:
            - box drops off the table
            - collision
        """
        box_pos = self.sim.data.body_xpos[self.box_body_id]
        table_height = self.model.mujoco_arena.table_offset[2]
        box_dropped = box_pos[2] <= table_height
        gripper_pos = [
            self.sim.data.qpos[x] for x in self.robots[0]._ref_gripper_joint_pos_indexes
        ]

        if box_dropped:
            return True

        # oracle to detect failures
        # TODO classifier
        # detects collisions
        is_collision, is_slip, is_missed_obj = False, False, False
        slip = self._get_observations()["slip"]

        if self._is_collision():
            logger.debug("  COLLISION")
            logger.debug(f"    slip: {slip}")
            logger.debug(f"    collision: {self.robots[0].ee_torque}")
            logger.debug(f"    collision sign: {np.sign(self.robots[0].ee_torque[0])}")
            if self.monitor_failure["collision"]:
                is_collision = True

        if slip != -1 and abs(slip) > self.reward_cfg.contact.slip_thresh:
            logger.debug("  SLIP")
            if self.monitor_failure["slip"]:
                is_slip = True

        if np.sum(np.abs(gripper_pos)) < 0.005:
            logger.debug("  MISSED_OBJ")
            if self.monitor_failure["missed_obj"]:
                is_missed_obj = True

        fail = None

        if is_collision and is_slip:
            fail = {
                "state": self.observe_true_state(),
                "type": "collision-slip",
                "stage": self.stage,
            }
        elif is_collision:
            fail = {
                "state": self.observe_true_state(),
                "type": "collision",
                "stage": self.stage,
            }
        elif is_slip:
            fail = {
                "state": self.observe_true_state(),
                "type": "slip",
                "stage": self.stage,
            }
        elif is_missed_obj:
            fail = {
                "state": self.observe_true_state(),
                "type": "missed_obj",
                "stage": self.stage,
            }

        if fail is None:
            return False
        else:
            self.failures.append(fail)
            return True

    def _get_placement_initializer(self):
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        shelf_sampler = QuatRandomSampler(
            name="ShelfSampler",
            mujoco_objects=self.shelf,
            # x_range=[0.05, 0.1],
            # y_range=[0.45, 0.5],
            # z_offset=np.random.uniform(0.4 + 0.075, 0.4 + 0.125),
            x_range=[0.0, 0.05],
            y_range=[-0.45, -0.40],
            # z_offset=0.05,
            z_offset=0.05,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
        )
        self.placement_initializer.append_sampler(shelf_sampler)
        box_sampler = UniformRandomSampler(
            name="BoxSampler",
            mujoco_objects=self.box,
            x_range=[0.0, 0.0],
            y_range=[-0.20, -0.20],
            rotation_axis="z",
            rotation=0,
            ensure_object_boundary_in_range=False,
            # ensure_valid_placement=True,
            ensure_valid_placement=False,
            reference_pos=self.table_offset,
            z_offset=0.0,
        )
        self.placement_initializer.append_sampler(box_sampler)

        if self.task_dist["clutter"]["enable"]:
            for i in range(self.task_dist["clutter"]["n_clutter"]):
                clutter_sampler = UniformRandomSampler(
                    name=f"ClutterSampler{i}",
                    mujoco_objects=self.clutters[i],
                    # x_range=[0.0 + 0.05*i, 0.0 + 0.05 *(i+1)],
                    x_range=[-0.1, 0.1],
                    y_range=[0.3, 0.35],
                    rotation_axis="z",
                    rotation=0,
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                    reference_pos=self.table_offset,
                    z_offset=0.01,
                    # z_offset=0.0,
                )
                self.placement_initializer.append_sampler(clutter_sampler)

    def _robot_grasping_box(self):
        gripper_geoms = [
            self.robots[0].gripper.important_geoms["left_finger"],
            self.robots[0].gripper.important_geoms["right_finger"],
        ]
        robot_grasping_box = all(
            [self.check_contact(geom, self.box) for geom in gripper_geoms]
        )
        return robot_grasping_box

    def _robot_touching_box(self):
        gripper_geoms = [
            self.robots[0].gripper.important_geoms["left_finger"],
            self.robots[0].gripper.important_geoms["right_finger"],
        ]
        robot_touching_box = any(
            [self.check_contact(geom, self.box) for geom in gripper_geoms]
        )
        return robot_touching_box

    # def _shelf_wall_plane(obs_cache):
    # normal = np.array([0, -1, 0])
    # point =

    def _box_bottom_plane(self):
        box_mat = T.quat2mat(self.sim.data.body_xquat[self.box_body_id])
        box_pos = self.sim.data.body_xpos[self.box_body_id]
        box_com_T = RT(box_mat, box_pos, from_frame="box")

        bottom_wrt_com = RT(
            rotation=np.eye(3),
            translation=self.box.bottom_offset,
            from_frame="bottom",
            to_frame="box",
        )

        bottom_T = box_com_T * bottom_wrt_com

        bottom_pos = bottom_T.translation

        # facing down
        bottom_vec_wrt_com = np.array([0, 0, -1])
        bottom_vec = np.dot(box_mat, bottom_vec_wrt_com)

        return bottom_vec, bottom_pos

    def _box_front_plane(self):
        box_mat = T.quat2mat(self.sim.data.body_xquat[self.box_body_id])
        box_pos = self.sim.data.body_xpos[self.box_body_id]
        box_com_T = RT(box_mat, box_pos, from_frame="box")

        front_wrt_com = RT(
            rotation=np.eye(3),
            translation=np.array([0, self.box.size[1], 0]),
            from_frame="front",
            to_frame="box",
        )

        front_T = box_com_T * front_wrt_com

        front_pos = front_T.translation

        # facing forward
        front_vec_wrt_com = np.array([0, 1, 0])
        front_vec = np.dot(box_mat, front_vec_wrt_com)

        return front_vec, front_pos

    def _table_plane(self):
        normal = np.array([0, 0, 1])
        point = deepcopy(self.table_offset)
        return normal, point

    def _shelf_plane(self):
        shelf_pos = deepcopy(self.sim.model.body_pos[self.shelf_body_id])
        shelf_wall_pos = shelf_pos
        shelf_wall_pos[1] += self.shelf.dims[1] / 2

        shelf_wall_vec = np.array([0, -1, 0])

        return shelf_wall_vec, shelf_wall_pos

    def _is_collision(self):
        eef_forces = self.robots[0].ee_force
        collision = np.max(eef_forces) > self.reward_cfg.eef_force.thresh
        return collision


class QuatRandomSampler(UniformRandomSampler):
    def _sample_quat(self):
        """
        Samples the orientation for a given object

        Returns:
            np.array: sampled (r,p,y) euler angle orientation

        Raises:
            ValueError: [Invalid rotation axis]
        """
        if self.rotation is None:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.rotation, collections.abc.Iterable):
            rot_angle = np.random.uniform(
                high=max(self.rotation), low=min(self.rotation)
            )
        else:
            rot_angle = self.rotation

        quat = T.mat2quat(T.euler2mat([-np.pi / 2, 0, 0]))
        return quat
