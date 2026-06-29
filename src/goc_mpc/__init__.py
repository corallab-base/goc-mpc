"""
Graph-of-Constraints Python Module
"""

__version__ = "0.9.2"

from enum import Enum

from .goc_mpc import (
    GraphOfConstraints,
    GraphOfConstraintsMPC,
    WaypointSolver,
    WaypointObjective,
    GraphWaypointMPC,
    GraphTimingMPC,
    GraphShortPathMPC,
)


class ObsType(Enum):
    SPHERE = "sphere"
    BOX = "box"
