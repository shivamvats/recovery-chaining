import numpy as np
from .shelf_env import ShelfEnv


class ShelfEnvV2(ShelfEnv):

    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init(*args, **kwargs)

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
            material=redwood,
            # margin=1,
            # gap=1,
            # density=100,
            density=np.random.uniform(
                self.task_dist["box_density"][0], self.task_dist["box_density"][1]
            ),
        )

        # Add sideview
        mujoco_arena.set_camera(
            camera_name="sideview",
            pos=[-0.05651774593317116, -1.2761224129427358, 1.4879572214102434],
            quat=[-0.8064181, -0.5912228, 0.00687796, 0.00990507],
        )

        self.shelf = Shelf(
            "shelf",
            bin_size=(
                0.3,
                np.random.uniform(
                    self.task_dist["shelf_height"][0], self.task_dist["shelf_height"][1]
                ),
                0.2,
            ),
            transparent_walls=True,
        )

        # Create placement initializer
        self._get_placement_initializer()

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.shelf, self.box],
        )

    def _setup_references(self):
        super()._setup_references()
        self.clutter_body_id = self.sim.model.body_name2id(self.clutter.root_body)
