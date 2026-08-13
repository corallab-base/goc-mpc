"""
Graph-of-Constraints Python Module
"""

__version__ = "0.9.2"

from .goc_mpc import (
    GraphOfConstraints,
    GraphOfConstraintsMPC,
    WaypointSolver,
    WaypointObjective,
    GraphWaypointMPC,
    MILPWaypointMPC,
    GraphTimingMPC,
    GraphShortPathMPC,
    AdmmShortPathMPC,
    ObstacleSet,
)
try:
    from .evolutionary_waypoint_solver import EvolutionaryWaypointSolver, GraphOrderingSpec
except ModuleNotFoundError:
    print("Unable to import EvolutionaryWaypointSolver")
