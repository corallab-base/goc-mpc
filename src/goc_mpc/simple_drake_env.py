from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from importlib.resources import files
from pathlib import Path

from pydrake.multibody.plant import MultibodyPlant, AddMultibodyPlantSceneGraph
from pydrake.multibody.tree import ModelInstanceIndex
from pydrake.multibody.parsing import Parser
from pydrake.geometry import HalfSpace, Box, Rgba, SceneGraph, StartMeshcat, MeshcatVisualizer
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.analysis import Simulator
from pydrake.math import RigidTransform
from pydrake.multibody.plant import CoulombFriction

import goc_mpc


def _add_ground(plant, z=0.0, size=200.0, thickness=0.02, mu_s=0.9, mu_d=0.5):
    # Collision: infinite plane with +z normal through (0,0,z)
    X_WG = HalfSpace.MakePose(np.array([0., 0., 1.]), np.array([0., 0., z]))
    plant.RegisterCollisionGeometry(
        plant.world_body(), X_WG, HalfSpace(), "ground_collision",
        CoulombFriction(mu_s, mu_d)
    )
    # Visual: big thin box, sunk slightly to avoid z-fighting with grid
    plant.RegisterVisualGeometry(
        plant.world_body(),
        RigidTransform([0., 0., z - thickness/2.0]),
        Box(size, size, thickness),
        "ground_visual",
        np.array([0.75, 0.75, 0.75, 1.0])
    )


class SimpleDrakeGym:
    """
    Physics on for everything; 'teleportation' (direct set of q/v) only for
    selected model instances (robots). Uncontrolled objects evolve via physics.
    Observation is always concat([q, v]) for the entire plant.

    step(action_q, action_v=None):
        - action_q is a single 1D np.ndarray formed by concatenating the q
          for each controlled model instance, in the order given at __init__.
        - action_v (optional) is the similarly-concatenated v vector.
          If None, velocities for controlled models are set to zeros.
        - After writing (q,v) for controlled instances, we AdvanceTo(t+dt).
    """

    def __init__(
        self,
        controlled_model_names: List[str],   # e.g. ["point_mass_1", ...]
        passive_model_names: List[str],      # e.g. ["cube_1", ...]
        dt: float = 1.0/60.0,
        open_browser: bool = False,
        meshcat=None,
    ):
        self._dt = float(dt)

        # --- Build plant + scene_graph together (ports are wired for geometry). ---
        builder = DiagramBuilder()
        self.plant, self._scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        parser = Parser(self.plant)

        # --- Resolve resource files in your package. ---
        # Robots:
        robot_path = Path(files(goc_mpc) / "descriptions" / "point_mass_3dof.urdf")
        if not robot_path.exists():
            raise FileNotFoundError(f"Robot file not found: {robot_path}")

        # Cubes: try a few common names; edit if yours differs.
        cube_candidates = [
            Path(files(goc_mpc) / "descriptions" / "cube_3dof.urdf"),
        ]
        cube_path: Optional[Path] = next((p for p in cube_candidates if p.exists()), None)
        if cube_path is None:
            raise FileNotFoundError(
                "Could not find a cube model under goc_mpc/descriptions/ "
                "(tried cube.sdf, unit_cube.sdf, cube.urdf)."
            )

        # --- Add model instances with the exact names provided. ---
        self._controlled_names = list(controlled_model_names)
        self._passive_names = list(passive_model_names)
        self._controlled = []
        self._passive = []

        # call this before Finalize()
        _add_ground(self.plant)

        for name in self._controlled_names:
            mis = parser.AddModels(str(robot_path))
            # assuming mis is just one model instance
            self.plant.RenameModelInstance(mis[0], name)
            self.plant.set_gravity_enabled(mis[0], False)
            self._controlled.extend(mis)

        for name in self._passive_names:
            mis = parser.AddModels(str(cube_path))
            self.plant.RenameModelInstance(mis[0], name)
            self._passive.extend(mis)

        # Finalize plant (geometry registration is already handled).
        self.plant.Finalize()

        # --- Meshcat visualizer ---
        # open_browser=open_browser
        self._meshcat = meshcat or StartMeshcat()
        MeshcatVisualizer.AddToBuilder(builder, self._scene_graph, self._meshcat)

        # --- Build diagram & contexts ---
        self._diagram = builder.Build()
        self._context = self._diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyMutableContextFromRoot(self._context)
        self._sim = Simulator(self._diagram, context=self._context)
        self._sim.set_publish_every_time_step(False)

        # --- Sizes for obs/action bookkeeping (same as before) ---
        self._nq = self.plant.num_positions()
        self._nv = self.plant.num_velocities()
        self._ctrl_nq = [self.plant.num_positions(mi) for mi in self._controlled]
        self._ctrl_nv = [self.plant.num_velocities(mi) for mi in self._controlled]
        self._ctrl_q_slices = self._compute_slices(self._ctrl_nq)
        self._ctrl_v_slices = self._compute_slices(self._ctrl_nv)
        self._action_q_size = sum(self._ctrl_nq)
        self._action_v_size = sum(self._ctrl_nv)

        # --- Place models along x-axis: robots @ z=1.0, cubes @ z=0.1. ---
        # We do it *in the context* so we don't remove any DOFs (no welding).
        # For free bodies we set a free-body pose; otherwise we set the first 3
        # joint positions (assumes point_mass_3dof is [x,y,z]).
        spacing = 1.0

        # Helpers
        def _place_free_or_q(mi, xyz):
            x, y, z = xyz
            # Try to set a free-body pose on a likely root body (named "anchor" if present).
            body = None
            try:
                body = self.plant.GetBodyByName("anchor", mi)
            except RuntimeError:
                # fall back to the first body of the instance
                body_indices = self.plant.GetBodyIndices(mi)
                if body_indices:
                    body = self.plant.get_body(body_indices[0])

            if body is not None:
                try:
                    self.plant.SetFreeBodyPose(self.plant_context, body, RigidTransform([x, y, z]))
                    return
                except Exception:
                    pass  # not a free body; fall through

            # Fallback: set first three positions if they exist (e.g., prismatic x,y,z)
            nq_i = self.plant.num_positions(mi)
            q_i = np.zeros(nq_i)
            if nq_i >= 3:
                q_i[:3] = [x, y, z]
            self.plant.SetPositions(self.plant_context, mi, q_i)

        for i, mi in enumerate(self._controlled):
            _place_free_or_q(mi, (i * spacing, 0.0, 1.0))

        for i, mi in enumerate(self._passive):
            _place_free_or_q(mi, (i * spacing, 0.0, 0.1))

        # --- Cache defaults & show initial frame ---
        self._q_default = self.plant.GetPositions(self.plant_context).copy()
        self._v_default = self.plant.GetVelocities(self.plant_context).copy()
        self._diagram.ForcedPublish(self._context)

    # ---- Gym-ish API ---------------------------------------------------------

    def reset(
        self,
        q0: Optional[np.ndarray] = None,
        v0: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset entire plant state (not just controlled). Returns (obs, info)."""
        q = self._q_default if q0 is None else q0
        v = self._v_default if v0 is None else v0
        if q.shape != (self._nq,) or v.shape != (self._nv,):
            raise ValueError(f"q must be ({self._nq},), v must be ({self._nv},)")
        self.plant.SetPositions(self.plant_context, q)
        self.plant.SetVelocities(self.plant_context, v)
        self._diagram.ForcedPublish(self._context)
        return self._observe(), {}

    def step(
        self,
        action_q: np.ndarray,
        action_v: Optional[np.ndarray] = None,
        *,
        reward: float = 0.0
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Write (q,v) for controlled models, then advance physics by dt.
        Uncontrolled models evolve due to contact/dynamics.
        """
        if action_q.shape != (self._action_q_size,):
            raise ValueError(f"action_q must be shape ({self._action_q_size},)")
        if action_v is not None and action_v.shape != (self._action_v_size,):
            raise ValueError(f"action_v must be shape ({self._action_v_size},)")

        # 1) Apply commanded q (and v) to each controlled model instance.
        self._apply_controlled_qv(action_q, action_v)

        # 2) Advance simulator so objects/contacts evolve.
        self._sim.AdvanceTo(self._context.get_time() + self._dt)

        obs = self._observe()
        done = False
        truncated = False
        info: Dict[str, Any] = {}
        return obs, reward, done, truncated, info

    def render(self):
        self._diagram.ForcedPublish(self._context)

    def close(self):
        try:
            self._meshcat.DeleteAddedControls()
        except Exception:
            pass

    # ---- Helpers -------------------------------------------------------------

    @property
    def action_q_size(self) -> int:
        """Total length of action_q (sum of q sizes across controlled models)."""
        return self._action_q_size

    @property
    def action_v_size(self) -> int:
        """Total length of action_v (sum of v sizes across controlled models)."""
        return self._action_v_size

    @property
    def nq(self) -> int:
        return self._nq

    @property
    def nv(self) -> int:
        return self._nv

    def pack_action(
        self,
        per_model_q: Dict[str, np.ndarray],
        per_model_v: Optional[Dict[str, np.ndarray]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Utility: build (action_q, action_v) from per-model arrays keyed by name,
        in the constructor-specified order of controlled models.
        """
        aq_list = []
        av_list = [] if per_model_v is not None else None
        for i, mi in enumerate(self._controlled):
            name = self.plant.GetModelInstanceName(mi)
            q_i = per_model_q[name]
            if q_i.shape != (self._ctrl_nq[i],):
                raise ValueError(f"{name}.q must be shape ({self._ctrl_nq[i]},)")
            aq_list.append(q_i)
            if av_list is not None:
                v_i = per_model_v.get(name, None)
                if v_i is None:
                    raise ValueError(f"Missing v for controlled model '{name}'")
                if v_i.shape != (self._ctrl_nv[i],):
                    raise ValueError(f"{name}.v must be shape ({self._ctrl_nv[i]},)")
                av_list.append(v_i)
        action_q = np.concatenate(aq_list) if aq_list else np.zeros(0)
        action_v = np.concatenate(av_list) if av_list is not None else None
        return action_q, action_v

    # Internal:

    def _observe(self) -> np.ndarray:
        q = self.plant.GetPositions(self.plant_context)
        v = self.plant.GetVelocities(self.plant_context)
        return np.concatenate([q, v])

    @staticmethod
    def _compute_slices(lengths: List[int]) -> List[slice]:
        """Return contiguous slices [0:n0), [n0:n0+n1), ... for given lengths."""
        s = 0
        out = []
        for L in lengths:
            out.append(slice(s, s + L))
            s += L
        return out

    def _apply_controlled_qv(self, action_q: np.ndarray, action_v: Optional[np.ndarray]):
        # For each controlled model, set q (and v) in its own coordinates.
        for i, mi in enumerate(self._controlled):
            q_slice = self._ctrl_q_slices[i]
            q_i = action_q[q_slice]
            self.plant.SetPositions(self.plant_context, mi, q_i)

            if action_v is not None:
                v_slice = self._ctrl_v_slices[i]
                v_i = action_v[v_slice]
            else:
                v_i = np.zeros(self._ctrl_nv[i])
            self.plant.SetVelocities(self.plant_context, mi, v_i)
