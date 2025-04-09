import logging
import numpy as np
from robosuite.environments.manipulation.pick_place import PickPlaceBread
import belief_srs.utils.transforms as T

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


class PickBread(PickPlaceBread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reward(self, action=None):
        rew = int(self._check_success())
        if self.reward_shaping:
            staged_rewards = self.staged_rewards()
            rew += max(staged_rewards)

        logger.debug(f"  Reward: {rew:.3f}")

        return rew

    def _check_success(self):
        """Check if the object has been lifted successfully."""
        obj_pos = self.sim.data.body_xpos[self.obj_body_id[self.obj_to_use]]
        target_height = 0.90
        if obj_pos[2] > target_height:
            return True
        else:
            return False

    def staged_rewards(self):
        """
        Returns staged rewards based on current physical states.
        Stages consist of reaching, grasping, lifting, and hovering.

        Returns:
            4-tuple:

                - (float) reaching reward
                - (float) grasping reward
                - (float) lifting reward
                - (float) hovering reward
        """

        reach_mult = 0.1
        grasp_mult = 0.35
        lift_mult = 0.5

        # filter out objects that are already in the correct bins
        active_objs = [self.objects[self.object_id]]

        # reaching reward governed by distance to closest object
        r_reach = 0.0
        # get reaching reward via minimum distance to a target object
        dists = [
            self._gripper_to_target(
                gripper=self.robots[0].gripper,
                target=active_obj.root_body,
                target_type="body",
                return_distance=True,
            )
            for active_obj in active_objs
        ]
        r_reach = (1 - np.tanh(10.0 * min(dists))) * reach_mult

        # grasping reward for touching any objects of interest
        r_grasp = (
            int(
                self._check_grasp(
                    gripper=self.robots[0].gripper,
                    object_geoms=[
                        g
                        for active_obj in active_objs
                        for g in active_obj.contact_geoms
                    ],
                )
            )
            * grasp_mult
        )

        # lifting reward for picking up an object
        r_lift = 0.0
        if active_objs and r_grasp > 0.0:
            z_target = self.bin2_pos[2] + 0.25
            object_z_locs = self.sim.data.body_xpos[
                [self.obj_body_id[active_obj.name] for active_obj in active_objs]
            ][:, 2]
            z_dists = np.maximum(z_target - object_z_locs, 0.0)
            r_lift = grasp_mult + (1 - np.tanh(15.0 * min(z_dists))) * (
                lift_mult - grasp_mult
            )

        logger.debug(f"  staged reward: {r_reach:.3f}, {r_grasp:.3f}, {r_lift:.3f}")

        return r_reach, r_grasp, r_lift
