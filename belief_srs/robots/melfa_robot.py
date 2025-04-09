import os

import numpy as np
from robosuite.models.robots.robot_model import register_robot
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.robots.single_arm import SingleArm



class MELFA(ManipulatorModel):

    def __init__(self, idn=0, duplicate_collision_geoms=True):
        pwd = os.path.dirname(os.path.realpath(__file__))
        super().__init__(os.path.join(pwd, "./melfa/melfa.xml"), idn=idn)

    @property
    def default_mount(self):
        #  return "RethinkMount"
        return None

    @property
    def default_gripper(self):
        # return "SchunkGripper"
        return "PandaGripper"

    @property
    def default_controller_config(self):
        return "default_melfa"

    @property
    def init_qpos(self):
        # return np.array([0.0, 0.8, 0.8, 0.0, np.pi / 2, 0.0])
        return np.array([1.05339361e-02, 4.46713539e-02, 1.43241821e+00,
                         2.90412568e-04, 1.69372694e+00, 1.05517502e-02])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0.5),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"

    @property
    def _eef_name(self):
        """
        XML eef name for this robot to which grippers can be attached. Note that these should be the raw
        string names directly pulled from a robot's corresponding XML file, NOT the adjusted name with an
        auto-generated naming prefix

        Returns:
            str: Raw XML eef name for this robot (default is "right_hand")
        """
        return "interface"


register_robot(MELFA)
ROBOT_CLASS_MAPPING["MELFA"] = SingleArm
