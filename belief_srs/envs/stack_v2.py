import numpy as np
from robosuite.environments.manipulation.stack import Stack
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.observables import *


class StackV2(Stack):
    """Stacking task."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_observables()
        self.record_failures = False
        self.failures = []
        self.transitions = []

    def _reset_interval(self):
        super()._reset_internal()

        self.failures, self.transitions = [], []

    def _pre_action(self, action, policy_step=False):
        # assume access to ground truth state
        self._pre_state = self.observe_true_state()
        self._action = action

        return super()._pre_action(action, policy_step)

    def _post_action(self, action):
        ret = super()._post_action(action)
        self._post_state = self.observe_true_state()

        # record failure transitions
        transition = (self._pre_state, self._action, self._post_state)
        self.transitions.append(transition)
        fail = self.is_fail_transition(transition)
        if fail:
            self.failures.append(transition)

        return ret

    def is_fail_transition(self, transition):
        """
        Evaluate whether a transition is safe and successful or not.

        1. Checks for eef forces.
        2. TODO: check for slip
        """
        FORCE_THRESH = 50

        pre, action, post = transition
        eef_forces = post['robot0_eef_forces']
        if np.max(eef_forces) > FORCE_THRESH:
            return True
        return False

    def observe_true_state(self):
        state = {
            "cubeA_pos": self.sim.data.body_xpos[self.cubeA_body_id],
            "cubeB_pos": self.sim.data.body_xpos[self.cubeB_body_id],
            "robot0_eef_pos": self.sim.data.site_xpos[
                self.sim.model.site_name2id("gripper0_grip_site")
            ],
            "robot0_eef_forces": self.robots[0].ee_force,
            "mj_state": self.sim.get_state().flatten()
        }
        return state

    def staged_rewards(self):
        # reaching is successful when the gripper site is close to the center of the cube
        cubeA_pos = self.sim.data.body_xpos[self.cubeA_body_id]
        cubeB_pos = self.sim.data.body_xpos[self.cubeB_body_id]
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        dist = np.linalg.norm(gripper_site_pos - cubeA_pos)
        r_reach = (1 - np.tanh(10.0 * dist)) * 0.25

        # grasping reward
        grasping_cubeA = self._check_grasp(
            gripper=self.robots[0].gripper, object_geoms=self.cubeA
        )
        if grasping_cubeA:
            r_reach += 0.25

        # lifting is successful when the cube is above the table top by a margin
        cubeA_height = cubeA_pos[2]
        table_height = self.table_offset[2]
        cubeA_lifted = cubeA_height > table_height + 0.04
        r_lift = 1.0 if cubeA_lifted else 0.0

        # Aligning is successful when cubeA is right above cubeB
        if cubeA_lifted:
            horiz_dist = np.linalg.norm(
                np.array(cubeA_pos[:2]) - np.array(cubeB_pos[:2])
            )
            r_lift += 0.5 * (1 - np.tanh(horiz_dist))

        # stacking is successful when the block is lifted and the gripper is not holding the object
        r_stack = 0
        cubeA_touching_cubeB = self.check_contact(self.cubeA, self.cubeB)
        gripper_geoms = [
            self.robots[0].gripper.important_geoms["left_finger"],
            self.robots[0].gripper.important_geoms["right_finger"],
        ]
        touching_cubeA = any(
            [self.check_contact(geom, self.cubeA) for geom in gripper_geoms]
        )
        if not touching_cubeA and r_lift > 0 and cubeA_touching_cubeB:
            r_stack = 2.0

        return r_reach, r_lift, r_stack

    def _load_model(self):
        env_cfg = self.env_configuration
        if isinstance(env_cfg, dict):
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
            greenwood = CustomMaterial(
                texture="WoodGreen",
                tex_name="greenwood",
                mat_name="greenwood_mat",
                tex_attrib=tex_attrib,
                mat_attrib=mat_attrib,
            )
            self.cubeA = BoxObject(
                name="cubeA",
                size_min=env_cfg["cubeA"]["size_min"],
                size_max=env_cfg["cubeA"]["size_max"],
                rgba=[1, 0, 0, 1],
                material=redwood,
            )
            self.cubeB = BoxObject(
                name="cubeB",
                size_min=env_cfg["cubeB"]["size_min"],
                size_max=env_cfg["cubeB"]["size_max"],
                rgba=[0, 1, 0, 1],
                material=greenwood,
            )
            cubes = [self.cubeA, self.cubeB]
            # Create placement initializer
            if self.placement_initializer is not None:
                self.placement_initializer.reset()
                self.placement_initializer.add_objects(cubes)
            else:
                self.placement_initializer = UniformRandomSampler(
                    name="ObjectSampler",
                    mujoco_objects=cubes,
                    x_range=[-0.08, 0.08],
                    y_range=[-0.08, 0.08],
                    rotation=None,
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    z_offset=0.01,
                )

            # task includes arena, robot, and objects of interest
            self.model = ManipulationTask(
                mujoco_arena=mujoco_arena,
                mujoco_robots=[robot.robot_model for robot in self.robots],
                mujoco_objects=cubes,
            )

        else:
            super()._load_model()

    def _setup_observables(self):
        observables = super()._setup_observables()

        corrupter = create_gaussian_noise_corrupter(
            mean=np.zeros(3), std=np.ones(3) * 0.01
        )
        observables["cubeA_pos"].set_corrupter(corrupter)

        corrupter = create_gaussian_noise_corrupter(
            mean=np.zeros(3), std=np.ones(3) * 0.01
        )
        observables["cubeB_pos"].set_corrupter(corrupter)

        return observables
