import logging
import numpy as np

from robosuite.environments.manipulation.lift import Lift
from robosuite.utils.observables import create_gaussian_noise_corrupter
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.observables import *

import belief_srs.utils.transforms as T

logger = logging.getLogger(__name__)
logger.setLevel("INFO")


class PlaceAtEdge(Lift):
    """
    The task is to place a block at the edge of a table under uncertain block
    and edge position estimation.

    The task is irrecoverable if the cube is dropped from the table.
    """

    def __init__(self, *args, monitor_failure=False, **kwargs):
        kwargs["table_full_size"] = (0.4, 0.4, 0.05)
        super().__init__(*args, **kwargs)

        self.monitor_failure = monitor_failure
        self.failures = []

    def reward(self, action=None):
        reward = 0.0

        cube_pos = self.sim.data.body_xpos[self.cube_body_id]
        table_edge_y = self.table_offset[1] + self.table_full_size[1] / 2
        table_height = self.model.mujoco_arena.table_offset[2]

        cube_on_table = cube_pos[2] >= table_height
        dist_bw_cube_and_edge = table_edge_y - cube_pos[1]

        gripper_geoms = [
            self.robots[0].gripper.important_geoms["left_finger"],
            self.robots[0].gripper.important_geoms["right_finger"],
        ]
        touching_cube = any(
            [self.check_contact(geom, self.cube) for geom in gripper_geoms]
        )

        if cube_on_table and not touching_cube:
            # greater than 1 if cube over the edge
            reward += 1 - dist_bw_cube_and_edge

        else:
            # dropped cube or still grasping
            pass

        return reward

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
        self.cube = BoxObject(
            name="cube",
            size_min=[0.040, 0.020, 0.020],
            size_max=[0.040, 0.022, 0.022],
            # size_min=[0.020, 0.040, 0.020],
            # size_max=[0.022, 0.042, 0.022],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube,
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
                # rotation=None,
                rotation=(-np.pi/10, np.pi/10),
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cube,
        )

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # corner-related observables
            @sensor(modality=modality)
            def target_pos(obs_cache):
                local_corner_pos = np.array(
                    [
                        0.0,
                        self.table_full_size[1] / 2,
                        self.table_full_size[2] / 2,
                    ]
                )
                return local_corner_pos + self.table_offset

            sensors = [target_pos]
            names = [s.__name__ for s in sensors]
            corrupter = create_gaussian_noise_corrupter(
                mean=np.zeros(3),
                std=np.array([0.01, 0.01, 0.0]),
            )

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    corrupter=corrupter,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:
            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                )

    def _post_action(self, action):
        """
        Terminate if absorbing state reached.

        Add additional info.
        """
        reward, done, info = super()._post_action(action)

        # early termination
        done = done or self._check_terminated()

        return reward, done, info

    def _check_success(self):
        """
        Check if cube has been placed at table edge

        """
        cube_pos = self.sim.data.body_xpos[self.cube_body_id]
        table_edge_y = self.table_offset[1] + self.table_full_size[1] / 2
        table_height = self.model.mujoco_arena.table_offset[2]

        cube_on_table = cube_pos[2] >= table_height
        dist_bw_cube_and_edge = np.abs(table_edge_y - cube_pos[1])

        # logger.info(f"cube on table: {cube_on_table}")
        # logger.info(f"dist bw cube and edge: {dist_bw_cube_and_edge}")

        gripper_geoms = [
            self.robots[0].gripper.important_geoms["left_finger"],
            self.robots[0].gripper.important_geoms["right_finger"],
        ]
        touching_cube = any(
            [self.check_contact(geom, self.cube) for geom in gripper_geoms]
        )

        # max cube half-size is 0.022
        return cube_on_table and not touching_cube and (dist_bw_cube_and_edge <= 0.025)

    def _check_terminated(self):
        """
        Terminate if:
            - cube drops
            - failure
        """
        cube_pos = self.sim.data.body_xpos[self.cube_body_id]
        table_height = self.model.mujoco_arena.table_offset[2]
        cube_on_table = cube_pos[2] >= table_height

        if not cube_on_table:
            return True

        if self.monitor_failure:
            # oracle to detect failures
            # TODO classifier
            # to be used during place action
            # after release, block should be stable and should not touch just
            # one finger
            left_finger = self.robots[0].gripper.important_geoms["left_finger"]
            right_finger = self.robots[0].gripper.important_geoms["right_finger"]
            touching_left = self.check_contact(left_finger, self.cube)
            touching_right = self.check_contact(right_finger, self.cube)
            unstable = touching_left != touching_right
            # logger.info(f"touching: {touching_left}, {touching_right}")
            if unstable:
                self.unstable_t += 1
                self.failures.append(self.observe_true_state())
                # self.failures.append({"state"": self.observe_true_state(),
                                      # "obs":})
            else:
                self.unstable_t = 0

            if self.unstable_t > 5:
                logger.debug("Failure detected! Terminating...")
                return True
            else:
                return False

        return False

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

    def observe_true_state(self):
        state = {
            "cube_pos": self.sim.data.body_xpos[self.cube_body_id],
            "cube_mat": T.quat2mat(self.sim.data.body_xquat[self.cube_body_id]),
            "robot0_eef_pos": self.sim.data.site_xpos[
                self.sim.model.site_name2id("gripper0_grip_site")
            ],
            "robot0_eef_mat": np.array(
                self.sim.data.site_xmat[
                    self.sim.model.site_name2id("gripper0_grip_site")
                ].reshape([3, 3])
            ),
            "robot0_gripper_pos": np.array(
                [
                    self.sim.data.qpos[x]
                    for x in self.robots[0]._ref_gripper_joint_pos_indexes
                ]
            ),
            "robot0_eef_forces": self.robots[0].ee_force,
            "obs": self._get_observations(force_update=False),
            "mj_state": self.sim.get_state().flatten(),
            "mj_xml": self.sim.model.get_xml(),
        }
        return state
