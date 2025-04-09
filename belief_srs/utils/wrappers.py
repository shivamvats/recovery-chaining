import cv2
from gymnasium import spaces
from gymnasium.core import ObservationWrapper, Wrapper
import matplotlib.pyplot as plt
import numpy as np
import time


class RGBImgObsWrapper(ObservationWrapper):
    """
    Wrapper to use fully/partially observable RGB image as observation,
    This can be used to have the agent to solve the gridworld in pixel space.

    It differs from minigrid.wrappers.ImgObsWrapper in that it supports not
    highlighting the agent's field of view and supports returning a partial
    view.

    Example:
        >>> import miniworld
        >>> import gymnasium as gym
        >>> import matplotlib.pyplot as plt
        >>> from minigrid.wrappers import RGBImgObsWrapper
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> obs, _ = env.reset()
        >>> plt.imshow(obs['image'])
        ![NoWrapper](../figures/lavacrossing_NoWrapper.png)
        >>> env = RGBImgObsWrapper(env)
        >>> obs, _ = env.reset()
        >>> plt.imshow(obs['image'])
        ![RGBImgObsWrapper](../figures/lavacrossing_RGBImgObsWrapper.png)
    """

    def __init__(self, env, tile_size=8, highlight=True, partial_view=False):
        super().__init__(env)

        self.tile_size = tile_size
        self.highlight = highlight
        self.partial_view = partial_view

        if partial_view:
            new_image_space = spaces.Box(
                low=0,
                high=255,
                shape=(
                    self.env.agent_view_size * tile_size,
                    self.env.agent_view_size * tile_size,
                    3,
                ),
                dtype="uint8",
            )

        else:
            new_image_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.env.width * tile_size, self.env.height * tile_size, 3),
                dtype="uint8",
            )

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def get_view_exts(self, agent_view_size=None):
        """
        Get the extents of the square set of tiles centered on the agent.
        if agent_view_size is None, use self.agent_view_size
        """

        agent_view_size = agent_view_size or self.agent_view_size

        topX = self.agent_pos[0] - agent_view_size // 2
        topY = self.agent_pos[1] - agent_view_size // 2
        botX = topX + agent_view_size
        botY = topY + agent_view_size

        return (topX, topY, botX, botY)

    def get_partial_render(self, highlight, tile_size):
        """
        Render a paratial observation.
        """
        agent_view_size = self.agent_view_size

        # generate sub-grid centered on agent
        # ------------------------------------
        topX, topY, botX, botY = self.get_view_exts(agent_view_size)

        grid = self.grid.slice(topX, topY, agent_view_size, agent_view_size)

        # for i in range(self.agent_dir + 1):
        # grid = grid.rotate_left()

        # Process occluders and visibility
        # Note that this incurs some performance cost
        vis_mask = np.ones(shape=(grid.width, grid.height), dtype=bool)

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        agent_pos = grid.width // 2, grid.height // 2
        if self.carrying:
            grid.set(*agent_pos, self.carrying)
        else:
            grid.set(*agent_pos, None)

        # Render the whole grid
        img = grid.render(
            tile_size,
            agent_pos=agent_pos,
            agent_dir=self.agent_dir,
            highlight_mask=vis_mask,
        )

        return img

    def get_frame(self, highlight, tile_size, partial_view):
        if partial_view:
            return self.get_partial_render(highlight, tile_size)
        else:
            return self.env.get_frame(highlight, tile_size)

    def observation(self, obs):
        rgb_img = self.get_frame(
            highlight=self.highlight,
            tile_size=self.tile_size,
            partial_view=self.partial_view,
        )

        return {**obs, "image": rgb_img}


class ProcessImageWrapper(ObservationWrapper):
    def __init__(self, env, norm_obs=True, resize_and_gray=True, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.norm_obs = norm_obs
        self.resize_and_gray = resize_and_gray
        self.size = 84

        if self.resize_and_gray:
            obs_space_shape = (self.size, self.size, 3)
        else:
            obs_space_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0 if norm_obs else 255.0,
            shape=(1 if resize_and_gray else 3, obs_space_shape[0], obs_space_shape[1]),
            dtype="float32",
        )

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(observation), reward, terminated, truncated, info

    def observation(self, observation):
        """Returns a modified observation.

        Args:
            observation: The :attr:`env` observation

        Returns:
            The modified observation
        """
        if self.resize_and_gray:
            observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
            observation = cv2.resize(
                observation, (self.size, self.size), interpolation=cv2.INTER_AREA
            )
        if self.norm_obs:
            observation = observation / 255

        if self.resize_and_gray:
            return np.expand_dims(observation, 0)
        else:
            return np.moveaxis(observation, 2, 0)
