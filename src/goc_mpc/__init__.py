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
    ObstacleSet,
)
try:
    from .evolutionary_waypoint_solver import EvolutionaryWaypointSolver, build_graph_ordering_problem
except ModuleNotFoundError:
    print("Unable to import EvolutionaryWaypointSolver")
