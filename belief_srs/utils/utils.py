import collections
from copy import copy
import logging
import os
import random
from typing import Iterator, Optional, Sequence

from hydra.utils import to_absolute_path
from skspatial.objects import Plane

import numpy as np
import pickle as pkl

logger = logging.getLogger(__name__)
# logger.setLevel("DEBUG")


def pkl_load(filename, hydra=False):
    if hydra:
        filename = to_absolute_path(filename)
    return pkl.load(open(filename, "rb"))


def pkl_dump(object, filename, hydra=False):
    if hydra:
        filename = to_absolute_path(filename)
    return pkl.dump(object, open(filename, "wb"))


def set_seed(seed):
    np.random.seed(seed)
    random.seed


def distance_from_plane_to_plane(normal1, point1, normal2, point2):
    """
    Returns:
        distance b/w point1 and the plane2
        angular distance b/w the normal of two planes
    """
    # want to be opposing
    ang_dist = 1 - np.dot(-normal1, normal2)
    logger.debug("  Plane to plane dist:")
    logger.debug(f"    angular: {ang_dist}")

    # shortest distance b/w point1 to the second plane
    plane = Plane(point=point2, normal=normal2)
    trans_dist = plane.distance_point(point1)
    logger.debug(f"    trans: {trans_dist}")

    return trans_dist, ang_dist
