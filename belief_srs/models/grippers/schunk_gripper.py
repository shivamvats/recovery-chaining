import os

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.models.grippers import GRIPPER_MAPPING, ALL_GRIPPERS


class SchunkGripper(GripperModel):

    def __init__(self, idn=0):
        pwd = os.path.dirname(os.path.realpath(__file__))
        super().__init__(os.path.join(pwd, "./schunk/schunk_gripper.xml"),
                         idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return None


GRIPPER_MAPPING["SchunkGripper"] = SchunkGripper
ALL_GRIPPERS = GRIPPER_MAPPING.keys()
