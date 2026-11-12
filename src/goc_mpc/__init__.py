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
    GCSWaypointMPC,
    SCIPWaypointMPC,
    LocalNLPSolver,
    InitializationOracle,
    InitializationRequest,
    AssignmentGuess,
    CandidateSet,
    InitialGuess,
    PropagationOracle,
    CollisionChecker,
    CollisionScene,
    CollisionResult,
    VampCollisionChecker,
    IkRequest,
    IkSolver,
    UrDhParameters,
    UrArm,
    UrAnalyticIK,
    ArcTravelTimeModel,
    SubtourElimination,
    GraphTimingMPC,
    GraphShortPathMPC,
    ObstacleSet,
    agent_link_names,
    agent_workspace_tracks,
)
from .fields import AgentField
try:
    from .evolutionary_waypoint_solver import EvolutionaryWaypointSolver, build_graph_ordering_problem
except ModuleNotFoundError:
    print("Unable to import EvolutionaryWaypointSolver")
