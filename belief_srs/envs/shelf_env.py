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
import belief_srs.utils.transforms as T
from belief_srs.utils.utils import *
from belief_srs.models.objects.shelf import Shelf
from belief_srs.models.grippers.schunk_gripper import SchunkGripper

logger = logging.getLogger(__name__)
# logger.setLevel("DEBUG")
# logger.setLevel("INFO")


class ShelfEnv(SingleArmEnv):
    """
    The task is to place a block at the edge of a table under uncertain block
    and edge position estimation.

    The task is irrecoverable if the cube is dropped from the table.
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
        monitor_failure=None,
        reward_cfg=None,
        task_dist=None,
        potential_fn=None,
        failure_detection=True,
    ):
        # table_full_size = (0.2, 0.2, 0.05) # until 17th aug
        table_full_size = (0.8, 0.8, 0.05)
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0.1, 0.8))
        self.task_dist = task_dist
        self.potential_fn = potential_fn
        self.failure_detection = failure_detection

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.reward_cfg = reward_cfg

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer
        self.clutter_body_ids = []
        self.n_fails = 0
        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

        if monitor_failure is None:
            self.monitor_failure = {
                "collision": True,  # by default
                "slip": False,
                "missed_obj": False,
            }
        else:
            self.monitor_failure = monitor_failure
        self.failure = None
        self._target_pos = np.array([0, 0, 0])
        self._box_quat_start = None
        self.clutters = []

    def _place_precondition_sat(self):
        box_pos = self.sim.data.body_xpos[self.box_body_id]
        box_quat = self.sim.data.body_xquat[self.box_body_id]
        q1 = pq.Quaternion(box_quat)
        q2 = pq.Quaternion(matrix=np.eye(3))
        rot_distance = pq.Quaternion.distance(q1, q2)
        # upright orientation
        if rot_distance > 0.5:
            return False

        # TODO do collision checking
        # poor man's collision checkng
        if box_pos[2] < self._target_pos[2]:  # and :
            return False

        return True

    def reward(self, action=None):
        reward = 0.0

        target_pos = self._target_pos
        box_pos = self.sim.data.body_xpos[self.box_body_id]
        box_quat = self.sim.data.body_xquat[self.box_body_id]

        q1 = pq.Quaternion(box_quat)
        q2 = pq.Quaternion(matrix=np.eye(3))
        rot_distance = pq.Quaternion.distance(q1, q2)

        if self.reward_shaping:
            if self.reward_cfg.distance.enabled:
                # goal reward
                # don't care about the exact x location as long as it is inside the shelf
                if self.reward_cfg.relax_target_pos:
                    dist_bw_box_and_target = np.linalg.norm(
                        target_pos[1:] - box_pos[1:]
                    )
                    # penalty if box outside shelf
                else:
                    dist_bw_box_and_target = np.linalg.norm(target_pos - box_pos)

                reward += 1 - dist_bw_box_and_target

                # encourage upright orientation
                reward += 0.1 * (1 - rot_distance)

            if self.reward_cfg.contact.enabled:
                # encourage exploring new contact modes
                # TODO encourage contact b/w all planes on the obj and all contact planes
                target_plane = self._table_plane()
                # target_plane = self._shelf_plane()
                obj_plane = self._box_bottom_plane()
                # obj_plane = self._box_front_plane()
                trans_dist, ang_dist = distance_from_plane_to_plane(
                    obj_plane[0], obj_plane[1], target_plane[0], target_plane[1]
                )
                contact_dist = 0
                if trans_dist < 0.1:
                    contact_dist += trans_dist
                    contact_dist += ang_dist
                    # logger.debug(f"    contact dist: {contact_dist}")
                    reward += self.reward_cfg.contact.coeff * contact_dist

        # Action costs and task constraints
        if self.reward_cfg.action.enabled:
            # constanct action cost
            reward_term = self.reward_cfg.action.coeff
            reward += reward_term

        if self.reward_cfg.eef_force.enabled:
            max_eef_force = max(
                # no penalty if force is below threshold
                0,
                max(abs(self.robots[0].ee_force)) - self.reward_cfg.eef_force.thresh,
            )
            reward_term = self.reward_cfg.eef_force.coeff * max_eef_force
            reward += reward_term
            # logger.debug(f"  eef force: {max_eef_force}, reward: {reward_term}")

        # logger.debug(f"  Total reward: {reward}")

        # prev: +bonus for achieving precondition+
        # if term_action_called and self.env._check_success():
        # now: bonus for reaching goal
        if self._check_success():
            reward += 1

        elif self._check_terminated():
            # no negative for failure makes Q value too optimistic
            # I want to differentiate b/w failure and not reaching goal
            # reward -= 1
            # I prefer 5 success 5 failures to 0 success no failures.
            reward += self.reward_cfg.fail_reward

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        self.robot_configs[0]["initial_qpos"] = np.array(
            [
                -0.37382664,
                1.20507675,
                -0.07829084,
                -1.44558628,
                1.19006665,
                1.41870704,
                1.22605236,
            ]
        )

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
            # size_min=[0.02, 0.06, 0.08],
            # size_max=[0.02, 0.08, 0.09],
            size_min=[0.02, 0.06, 0.08],
            size_max=[0.02, 0.06, 0.08],
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
                    size_min=[0.02, 0.05, 0.06],
                    size_max=[0.02, 0.06, 0.07],
                    rgba=[1, 0, 0, 1],
                    material=bluewood,
                    density=100,
                )
                self.clutters.append(clutter)

        # Add sideview
        mujoco_arena.set_camera(
            camera_name="sideview",
            # pos=[-0.05651774593317116, -1.2761224129427358, 1.4879572214102434],
            pos=[-0.05651774593317116, -1.2761224129427358, 1.0],
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

    def _setup_references(self):
        super()._setup_references()
        self.box_body_id = self.sim.model.body_name2id(self.box.root_body)
        self.shelf_body_id = self.sim.model.body_name2id(self.shelf.root_body)
        if self.task_dist["clutter"]["enable"]:
            self.clutter_body_ids = [
                self.sim.model.body_name2id(clutter.root_body)
                for clutter in self.clutters
            ]

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
            return target_pos

        @sensor(modality=modality)
        def box_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.box_body_id])

        @sensor(modality=modality)
        def box_dims(obs_cache):
            return np.array([0.02, 0.06, 0.8])

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
            contacts = self._obj_contacts()
            # print(contacts)
            # print("eef: ", np.around(self.robots[0].ee_force, 3), contacts)

            return contacts

        @sensor(modality=modality)
        def collision(obs_cache):
            return np.array([self._is_collision()], dtype=int)

        @sensor(modality=modality)
        def table_point(obs_cache):
            return deepcopy(self.table_offset)

        def clutter_pos(clutter_id):
            @sensor(modality=modality)
            def _clutter_pos(obs_cache):
                clutter_pos = self.sim.data.body_xpos[clutter_id]
                return deepcopy(clutter_pos)

            return _clutter_pos

        def clutter_rpy(clutter_id):
            @sensor(modality=modality)
            def _clutter_rpy(obs_cache):
                clutter_quat = self.sim.data.body_xquat[clutter_id]
                clutter_rpy = deepcopy(T.quat2euler(clutter_quat))
                return clutter_rpy

            return _clutter_rpy

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

        for i, clutter_id in enumerate(self.clutter_body_ids):
            sensors += [clutter_pos(clutter_id), clutter_rpy(clutter_id)]
            names += [f"clutter{i}_pos", f"clutter{i}_rpy"]

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
                    # corrupter=corrupter,
                    corrupter=None,
                    sampling_rate=self.control_freq,
                )
            elif name == "box_pos":
                corrupter = create_gaussian_noise_corrupter(
                    mean=np.zeros(3),
                    std=np.array([0.0, 0.01, 0.02]),
                )
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    corrupter=corrupter,
                    # corrupter=None,
                    # sampling_rate=self.control_freq,
                    sampling_rate=1,  # slow updates
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
        self.failure = None

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
            # target_pos[2] += (0.08 - 0.15)
            # print(f"shelf dims: {self.shelf.dims[2]}")
            target_pos[2] += 0.015 + self.box.top_offset[2] - self.shelf.dims[2] / 2
            target_pos[1] -= 0.05
            self._target_pos = target_pos

        self._box_quat_start = None

    def _pre_action(self, *args, **kwargs):
        super()._pre_action(*args, **kwargs)
        self._prev_state = self.observe_true_state()

    def _post_action(self, action):
        """
        Terminate if absorbing state reached.

        Add additional info.
        """
        reward, done, info = super()._post_action(action)

        is_success = self._check_success()
        info["is_success"] = is_success

        # TODO move this to failure detector wrapper as in pick_place
        if self._check_terminated():
            info["is_failure"] = True
            info["state"] = self.observe_true_state()
        else:
            info["is_failure"] = False

        # early termination
        # XXX if I terminate on first success, then the robot won't retract
        # terminate if success AND hand retracted
        done = done or self._check_terminated() or is_success

        # track slip
        # gripper closed
        if self._robot_grasping_box():
            if self._box_quat_start is None:
                # start tracking
                self._box_quat_start = deepcopy(
                    self.sim.data.body_xquat[self.box_body_id]
                )
                logger.debug(
                    f"Start tracking slip. box_quat_start: {self._box_quat_start}"
                )
        else:
            self._box_quat_start = None
        # logger.info(f"  Box quat: {self._box_quat_start}")

        return reward, done, info

    def _check_success(self):
        """
        Check if box has been placed at the target pos.

        """
        gripper_geoms = [
            self.robots[0].gripper.important_geoms["left_finger"],
            self.robots[0].gripper.important_geoms["right_finger"],
        ]
        target_pos = self._target_pos
        eef_pos = self.sim.data.site_xpos[
            self.sim.model.site_name2id("gripper0_grip_site")
        ]
        box_pos = self.sim.data.body_xpos[self.box_body_id]
        box_quat = self.sim.data.body_xquat[self.box_body_id]
        shelf_dims = self.shelf.dims
        shelf_pos = self.sim.model.body_pos[self.shelf_body_id]

        robot_touching_box = any(
            [self.check_contact(geom, self.box) for geom in gripper_geoms]
        )
        box_touching_shelf = self.check_contact(self.box, self.shelf)
        displace_bw_box_and_target = np.abs(target_pos - box_pos)
        # dist_bw_box_and_target = np.linalg.norm(target_pos - box_pos)
        q1 = pq.Quaternion(box_quat)
        q2 = pq.Quaternion(matrix=np.eye(3))
        rot_distance = pq.Quaternion.distance(q1, q2)

        # print("box x: ", box_pos[0])
        # print("Shelf base range: ", shelf_pos[0] - shelf_dims[0] / 2,
        # shelf_pos[0] + shelf_dims[0] / 2)
        if box_touching_shelf and not robot_touching_box:
            check_retraction = True
            if check_retraction:
                dist_bw_eef_and_box = np.linalg.norm(eef_pos - box_pos)
                if dist_bw_eef_and_box < 0.2:
                    return False
            if rot_distance < 0.1 and (
                displace_bw_box_and_target[1] < 0.05
                and displace_bw_box_and_target[2] < 0.05
            ):
                if self.reward_cfg.relax_target_pos:
                    if box_pos[0] < shelf_pos[0] + shelf_dims[0] / 2 and (
                        box_pos[0] > shelf_pos[0] - shelf_dims[0] / 2
                    ):
                        return True
                else:
                    if displace_bw_box_and_target[1] < 0.05:
                        return True

        return False

    def _check_terminated(self):
        """
        Terminate if:
            - box drops off the table
            - collision
        """
        if not self.failure_detection:
            return False

        box_pos = self.sim.data.body_xpos[self.box_body_id]
        table_height = self.model.mujoco_arena.table_offset[2]
        box_dropped = box_pos[2] <= (table_height + 0.03)
        # logger.info(f"  box height: {box_pos[2]}, table_height: {table_height}")
        gripper_pos = [
            self.sim.data.qpos[x] for x in self.robots[0]._ref_gripper_joint_pos_indexes
        ]

        if box_dropped:
            return True

        if self.clutters and self.reward_cfg.clutter_rot_max is not None:
            for clutter_id in self.clutter_body_ids:
                clutter_quat = self.sim.data.body_xquat[clutter_id]
                q1 = pq.Quaternion(clutter_quat)
                q2 = pq.Quaternion(matrix=np.eye(3))
                rot_distance = pq.Quaternion.distance(q1, q2)
                # print(f"  clutter rot: {rot_distance}, max rot: {self.reward_cfg.clutter_rot_max}")
                if rot_distance > self.reward_cfg.clutter_rot_max:
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
            # logger.info(f"  contact force: {np.max(self.robots[0].ee_force)}")
            if self.monitor_failure["collision"] or (
                # hard constraint on eef force
                np.max(self.robots[0].ee_force)
                > self.reward_cfg.eef_force.hard_thresh
            ):
                # logger.info("Collision!")
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
            # self.failures.append(fail)
            self.failure = fail
            self.n_fails += 1
            # logger.info(f"FAILURES: {self.n_fails}")
            return True

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        # TODO viz current belief of the robot
        # if vis_settings["grippers"]:
        # self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cube)

    def set_target_pos(self, pos):
        self._target_pos = pos

    def set_box_quat_start(self, quat):
        self._box_quat_start = quat

    def set_stage(self, i):
        self.stage = i

    def observe_true_state(self, obs=None, mj_state=True):
        box_pos = deepcopy(self.sim.data.body_xpos[self.box_body_id])
        box_mat = deepcopy(T.quat2mat(self.sim.data.body_xquat[self.box_body_id]))
        robot0_eef_mat = deepcopy(
            np.array(
                self.sim.data.site_xmat[
                    self.sim.model.site_name2id("gripper0_grip_site")
                ].reshape([3, 3])
            )
        )
        robot0_eef_pos = deepcopy(
            self.sim.data.site_xpos[self.sim.model.site_name2id("gripper0_grip_site")]
        )
        shelf_pos = (deepcopy(self.sim.data.body_xpos[self.shelf_body_id]),)

        eef_forces = self.robots[0].ee_force
        hard_collision = np.max(eef_forces) > self.reward_cfg.eef_force.thresh

        state = {
            "box_pos": box_pos,
            "box_mat": box_mat,
            "box_rpy": T.mat2euler(box_mat),
            "box_dims": deepcopy(np.array(self.box.size)),
            "shelf_pos": shelf_pos,
            "robot0_eef_to_shelf": shelf_pos - robot0_eef_pos,
            "shelf_dims": deepcopy(np.array(self.shelf.dims)),
            "target_pos": deepcopy(self._target_pos),
            "box_quat_start": deepcopy(self._box_quat_start),
            "robot0_eef_pos": robot0_eef_pos,
            "robot0_eef_mat": robot0_eef_mat,
            "robot0_eef_rpy": T.mat2euler(robot0_eef_mat),
            "robot0_gripper_pos": deepcopy(
                np.array(
                    [
                        self.sim.data.qpos[x]
                        for x in self.robots[0]._ref_gripper_joint_pos_indexes
                    ]
                )
            ),
            "robot0_eef_to_box": box_pos - robot0_eef_pos,
            "robot0_eef_forces": deepcopy(self.robots[0].ee_force),
            "collision": np.array([self._is_collision()], dtype=int),
            "hard_collision": np.array([hard_collision], dtype=int),
            "obj_contacts": deepcopy(self._obj_contacts()),
            "robot_grasping_box": np.array([self._robot_grasping_box()], dtype=int),
            "robot_touching_box": np.array([self._robot_touching_box()], dtype=int),
            "obs": self._get_observations(force_update=False) if obs is None else obs,
        }
        if self.clutters:
            for clutter_id, clutter in zip(self.clutter_body_ids, self.clutters):
                clutter_quat = self.sim.data.body_xquat[clutter_id]
                clutter_pos = self.sim.data.body_xpos[clutter_id]
                clutter_to_box = state["box_pos"] - clutter_pos
                state[clutter._name + "_pos"] = deepcopy(clutter_pos)
                state[clutter._name + "_quat"] = deepcopy(clutter_quat)
                state[clutter._name + "_rpy"] = deepcopy(T.quat2euler(clutter_quat))
                state[clutter._name + "_to_box"] = clutter_to_box
                state["robot0_eef_to_" + clutter._name] = clutter_pos - robot0_eef_pos

        if mj_state:
            state["mj_state"] = self.sim.get_state().flatten()
            state["mj_xml"] = self.sim.model.get_xml()

        return state

    def _get_placement_initializer(self):
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        shelf_z_offset = np.random.uniform(0.05, 0.10)
        shelf_sampler = QuatRandomSampler(
            name="ShelfSampler",
            mujoco_objects=self.shelf,
            # x_range=[0.05, 0.1],
            # y_range=[0.45, 0.5],
            # z_offset=np.random.uniform(0.4 + 0.075, 0.4 + 0.125),
            x_range=[0.0, 0.05],
            y_range=[0.05 + 0.25, 0.05 + 0.3],
            # z_offset=0.05,
            z_offset=shelf_z_offset,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
        )
        self.placement_initializer.append_sampler(shelf_sampler)
        box_sampler = UniformRandomSampler(
            name="BoxSampler",
            mujoco_objects=self.box,
            x_range=[0.0, 0.0],
            y_range=[0.05, 0.05],
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
                    z_offset=0.02 + shelf_z_offset,
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

    def _obj_contacts(self):
        include_geoms = [
            "shelf_base",
            "shelf_wall0",
            "shelf_wall1",
            "shelf_wall2",
            "shelf_wall3",
            "table_collision",
        ]
        for clutter in self.clutters:
            include_geoms.append(clutter._name + "_g0")

        include_geom_ids = [self.sim.model.geom_name2id(geom) for geom in include_geoms]
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

        quat = T.mat2quat(T.euler2mat([np.pi / 2, 0, 0]))
        return quat
