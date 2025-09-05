from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from importlib.resources import files
from pathlib import Path
from dataclasses import dataclass

from pydrake.common.eigen_geometry import Quaternion
from pydrake.multibody.plant import MultibodyPlant_, AddMultibodyPlantSceneGraph
from pydrake.multibody.plant import CoulombFriction
from pydrake.multibody.tree import ModelInstanceIndex, Body, Frame
from pydrake.multibody.parsing import Parser
from pydrake.multibody.math import SpatialVelocity
from pydrake.geometry import HalfSpace, Box, Rgba, SceneGraph, StartMeshcat, MeshcatVisualizer
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.analysis import Simulator
from pydrake.math import RollPitchYaw, RigidTransform, RotationMatrix

import corallab_assets
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

# def add_free_base_joint_for_object(plant, model_instance, base_body_name="anchor"):
#     """Ensure the object’s base is free-floating so SetFreeBodyPose works."""
#     Bo = plant.GetBodyByName(base_body_name, model_instance).body_frame()
#     # If the base is already connected by any joint, adding another will throw.
#     # So only do this for single-link “object” models that have no base joint.
#     plant.AddJoint(
#         FreeJoint(f"{plant.GetModelInstanceName(model_instance)}_free",
#                   plant.world_frame(), Bo, RigidTransform()))


@dataclass
class _Grasp:
    robot_name: str
    object_name: str
    robot_frame: Frame        # grasp frame on robot (e.g., "anchor" or "tool0")
    object_body: Body         # base body of the object
    X_Ro: RigidTransform      # fixed offset (robot grasp frame → object base)
    obj_model_instance: ModelInstanceIndex


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

        self._grasps: Dict[str, _Grasp] = {}   # keyed by object_name

        # --- Build plant + scene_graph together (ports are wired for geometry). ---
        builder = DiagramBuilder()
        self.plant, self._scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        parser = Parser(self.plant)

        # --- Resolve resource files in your package. ---
        # Robots:
        robot_path = Path(files(goc_mpc) / "descriptions" / "point_mass_3dof.urdf")
        if not robot_path.exists():
            raise FileNotFoundError(f"Robot file not found: {robot_path}")

        free_body_robot_path = Path(files(goc_mpc) / "descriptions" / "free_body_6dof.urdf")
        if not free_body_robot_path.exists():
            raise FileNotFoundError(f"Robot file not found: {free_body_robot_path}")

        ur5e_robot_path = Path(files(corallab_assets) / "ur5e" / "ur5e.urdf")
        if not ur5e_robot_path.exists():
            raise FileNotFoundError(f"Robot file not found: {ur5e_robot_path}")

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
            if "point_mass" in name:
                mis = parser.AddModels(str(robot_path))
            elif "free_body" in name:
                mis = parser.AddModels(str(free_body_robot_path))
            elif "ur5e" in name:
                mis = parser.AddModels(str(ur5e_robot_path))
            else:
                raise NotImplementedError("Currently implemented robots are point_mass and free_body.")

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
        self._passive_nq = [self.plant.num_positions(mi) for mi in self._passive]
        self._passive_nv = [self.plant.num_velocities(mi) for mi in self._passive]
        self._ctrl_nq = [self._get_q_dim(name) for name in self._controlled_names]
        self._ctrl_nv = [self._get_qdot_dim(name) for name in self._controlled_names]
        self._ctrl_q_slices = self._compute_slices(self._ctrl_nq)
        self._ctrl_v_slices = self._compute_slices(self._ctrl_nv)
        self._nq = sum(self._passive_nq) + sum(self._ctrl_nq)
        self._nv = sum(self._passive_nv) + sum(self._ctrl_nv)
        self._action_q_size = sum(self._ctrl_nq)
        self._action_v_size = sum(self._ctrl_nv)
        self._passive_q_slices = self._compute_slices(self._passive_nq, start=self._action_q_size)
        self._passive_v_slices = self._compute_slices(self._passive_nv, start=self._action_v_size)

        # --- Place models along x-axis: robots @ z=1.0, cubes @ z=0.1. ---
        # We do it *in the context* so we don't remove any DOFs (no welding).
        # For free bodies we set a free-body pose; otherwise we set the first 3
        # joint positions (assumes point_mass_3dof is [x,y,z]).
        spacing = 1.0

        # Helpers
        def _place_free_or_q(mi, xyz, quat=[1.0, 0.0, 0.0, 0.0]):
            x, y, z = xyz
            matrix = RotationMatrix(Quaternion(quat))
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
                    self.plant.SetFreeBodyPose(self.plant_context, body, RigidTransform(matrix, [x, y, z]))
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
            _place_free_or_q(mi, (i * spacing + 0.5, 0.1, 1.0), quat=[0.0, 0.0, 1.0, 0.0])

        for i, mi in enumerate(self._passive):
            _place_free_or_q(mi, (i * spacing + 0.25, 0.5, 0.1), quat=[1.0, 0.0, 0.0, 0.0])

        # --- Cache defaults & show initial frame ---
        self._q_default = self._get_q()
        self._v_default = self._get_qdot()
        self._diagram.ForcedPublish(self._context)

    def _get_model_q(self, name):
        mi = self.plant.GetModelInstanceByName(name)
        if "free_body" in name:
            body = self.plant.GetBodyByName("ee_link", mi)
            pose = self.plant.EvalBodyPoseInWorld(self.plant_context, body)
            return np.concatenate((pose.translation(), pose.rotation().ToQuaternion().wxyz()))
        elif "cube" in name:
            return self.plant.GetPositions(self.plant_context, mi)
        else:
            q = []
            joint_indices = self.plant.GetActuatedJointIndices(mi)
            for ji in joint_indices:
                joint = self.plant.get_joint(ji)
                q.append(joint.GetOnePosition(self.plant_context))
            return np.array(q)

    def _get_model_qdot(self, name):
        mi = self.plant.GetModelInstanceByName(name)
        if "free_body" in name:
            body = self.plant.GetBodyByName("ee_link", mi)
            spatial_velocity = self.plant.EvalBodySpatialVelocityInWorld(self.plant_context, body)
            return np.concatenate((spatial_velocity.translational(), spatial_velocity.rotational()))
        elif "cube" in name:
            return self.plant.GetVelocities(self.plant_context, mi)
        else:
            qdot = []
            joint_indices = self.plant.GetActuatedJointIndices(mi)
            for ji in joint_indices:
                joint = self.plant.get_joint(ji)
                q.append(joint.GetOneVelocity(self.plant_context))
            return np.array(qdot)

    def _get_q(self):
        q = []
        for name in self._controlled_names:
            q.append(self._get_model_q(name))
        for name in self._passive_names:
            q.append(self._get_model_q(name))
        return np.concatenate(q)

    def _get_qdot(self):
        qdot = []
        for name in self._controlled_names:
            qdot.append(self._get_model_qdot(name))
        for name in self._passive_names:
            qdot.append(self._get_model_qdot(name))
        return np.concatenate(qdot)

    def _get_q_dim(self, name):
        if "free_body" in name:
            return 7
        else:
            mi = self.plant.GetModelInstanceByName(name)
            return self.plant.get_actuated_dofs(mi)

    def _get_qdot_dim(self, name):
        if "free_body" in name:
            return 6
        else:
            mi = self.plant.GetModelInstanceByName(name)
            return self.plant.get_actuated_dofs(mi)

    def _set_q(self, q: np.ndarray):
        for i, name in enumerate(self._controlled_names):
            q_slice = self._ctrl_q_slices[i]
            self._set_model_q(name, q[q_slice])
        for i, name in enumerate(self._passive_names):
            q_slice = self._passive_q_slices[i]
            self._set_model_q(name, q[q_slice])

    def _set_qdot(self, qdot: np.ndarray):
        for i, name in enumerate(self._controlled_names):
            qdot_slice = self._ctrl_v_slices[i]
            self._set_model_qdot(name, qdot[qdot_slice])
        for i, name in enumerate(self._passive_names):
            qdot_slice = self._passive_v_slices[i]
            self._set_model_qdot(name, qdot[qdot_slice])

    def _set_model_q(self, name, q):
        mi = self.plant.GetModelInstanceByName(name)
        if "free_body" in name:
            # Interpret q as [x y z w x y z]
            x_W = q[:3]
            q_W = Quaternion(q[3:])

            # Pick the actual free base body in this model instance.
            body = self.plant.GetBodyByName("ee_link", mi)

            X_WB = RigidTransform(q_W, x_W)
            self.plant.SetFreeBodyPose(self.plant_context, body, X_WB)
        elif "cube" in name:
            return self.plant.SetPositions(self.plant_context, mi, q)
        else:
            joint_indices = self.plant.GetActuatedJointIndices(mi)
            for i, ji in enumerate(joint_indices):
                joint = self.plant.get_joint(ji)
                joint.SetPositions(self.plant_context, q[i:i+1])

    def _set_model_qdot(self, name, qdot):
        mi = self.plant.GetModelInstanceByName(name)

        if "free_body" in name:
            # Interpret qdot as [v_W, ω_W]; never convert ω→rpẏ
            v_W = qdot[:3]
            w_W = qdot[3:]

            # Pick the actual free base body in this model instance.
            body = self.plant.GetBodyByName("ee_link", mi)

            V_WB = SpatialVelocity(w_W, v_W) # expressed in World
            self.plant.SetFreeBodySpatialVelocity(body, V_WB, self.plant_context)
        elif "cube" in name:
            return self.plant.SetVelocities(self.plant_context, mi, qdot)
        else:
            joint_indices = self.plant.GetActuatedJointIndices(mi)
            for i, ji in enumerate(joint_indices):
                joint = self.plant.get_joint(ji)
                joint.SetVelocities(self.plant_context, q[i:i+1])

    def _set_controlled_q(self, q: np.ndarray):
        for i, name in enumerate(self._controlled_names):
            q_slice = self._ctrl_q_slices[i]
            self._set_model_q(name, q[q_slice])

    def _set_controlled_qdot(self, qdot: np.ndarray):
        for i, name in enumerate(self._controlled_names):
            qdot_slice = self._ctrl_v_slices[i]
            self._set_model_qdot(name, qdot[qdot_slice])

    # def _apply_controlled_qv(self, action_q: np.ndarray, action_v: Optional[np.ndarray]):
    #     # For each controlled model, set q (and v) in its own coordinates.
    #     for i, mi in enumerate(self._controlled):
    #         q_slice = self._ctrl_q_slices[i]
    #         q_i = action_q[q_slice]
    #         self.plant.SetPositions(self.plant_context, mi, q_i)
    #         if action_v is not None:
    #             v_slice = self._ctrl_v_slices[i]
    #             v_i = action_v[v_slice]
    #         else:
    #             v_i = np.zeros(self._ctrl_nv[i])
    #         self.plant.SetVelocities(self.plant_context, mi, v_i)


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
        self._set_q(q)
        self._set_qdot(v)
        self._grasps = {}
        self._diagram.ForcedPublish(self._context)
        return self._observe(), {}

    def activate_grasp(
        self,
        robot_name: str,
        object_name: str,
        grasp_frame_on_robot: str = "ee_link",
        object_base_body: str = "cb_body",
    ) -> None:
        """Start rigidly attaching `object_name` to `robot_name` at the current pose."""
        # Resolve instances & frames/bodies
        mi_r = self.plant.GetModelInstanceByName(robot_name)
        mi_o = self.plant.GetModelInstanceByName(object_name)
        R = self.plant.GetFrameByName(grasp_frame_on_robot, mi_r)
        Bo = self.plant.GetBodyByName(object_base_body, mi_o)

        # Compute current world poses
        X_WR = self.plant.CalcRelativeTransform(self.plant_context, self.plant.world_frame(), R)
        X_WO = self.plant.EvalBodyPoseInWorld(self.plant_context, Bo)
        # Cache relative pose robot→object
        X_Ro = X_WR.inverse().multiply(X_WO)   # X_Ro = X_WR⁻¹ * X_WO

        # Optional: make object kinematic-ish while grasped
        try:
            self.plant.set_gravity_enabled(mi_o, False)
        except Exception:
            pass

        self._grasps[object_name] = _Grasp(
            robot_name=robot_name,
            object_name=object_name,
            robot_frame=R,
            object_body=Bo,
            X_Ro=X_Ro,
            obj_model_instance=mi_o,
        )

    def release_grasp(self, object_name: str) -> None:
        """Stop rigid attachment for this object; it resumes normal physics."""
        g = self._grasps.pop(object_name, None)
        if g is not None:
            try:
                self.plant.set_gravity_enabled(g.obj_model_instance, True)
            except Exception:
                pass

    def step(
            self,
            action_q: np.ndarray,
            action_v: Optional[np.ndarray] = None,
            *,
            reward: float = 0.0,
            grasp_cmds: Optional[Tuple[Tuple[str, str, str], ...]] = None,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        grasp_cmds: optional tuple of commands, each as (cmd, robot_name, object_name)
          - cmd == "grab": activate grasp for (robot_name, object_name)
          - cmd == "release": release grasp for object_name (robot_name ignored)
        """
        # Apply any grasp commands first (use current state to capture X_Ro)
        if grasp_cmds:
            for (cmd, rname, oname) in grasp_cmds:
                if cmd == "grab":
                    self.activate_grasp(rname, oname)
                elif cmd == "release":
                    self.release_grasp(oname)

        # 1) Apply commanded q/v to controlled robots (your existing code):
        if action_q.shape != (self._action_q_size,):
            raise ValueError(f"action_q must be shape ({self._action_q_size},)")
        if action_v is not None and action_v.shape != (self._action_v_size,):
            raise ValueError(f"action_v must be shape ({self._action_v_size},)")

        self._set_controlled_q(action_q)
        if action_v is None:
            action_v = np.zeros((self._action_v_size,))
        self._set_controlled_qdot(action_v)

        # 2) Enforce grasps *before* physics step (prevents big impulses)
        self._enforce_all_grasps_(match_velocity=True)

        # 3) Advance physics
        t_next = self._context.get_time() + self._dt
        self._sim.AdvanceTo(t_next)

        # 4) Enforce again post-step (eliminates small integration drift)
        self._enforce_all_grasps_(match_velocity=True)

        obs = self._observe()
        return obs, reward, False, False, {}

    def _set_cube_xyz_fast(self, cube_name: str, p_WO):
        """p_WO = (x, y, z) world position; assumes q[:3] = [x,y,z]."""
        mi = self.plant.GetModelInstanceByName(cube_name)
        q = self.plant.GetPositions(self.plant_context, mi).copy()
        q[:3] = p_WO
        self.plant.SetPositionsForModelInstance(self.plant_context, mi, q)

    # def _set_cube_vxyz_fast(self, cube_name: str, v_WO):
    #     """v_WO = (vx,vy,vz) world linear velocity; assumes v[:3] = [vx,vy,vz]."""
    #     mi = self.plant.GetModelInstanceByName(cube_name)
    #     v = self.plant.GetVelocities(self.plant_context, mi).copy()
    #     v[:3] = v_WO
    #     self.plant.SetVelocitiesForModelInstance(self.plant_context, mi, v)

    def _enforce_all_grasps_(self, match_velocity: bool = True) -> None:
        """Make each grasped object follow its robot grasp frame exactly."""
        W = self.plant.world_frame()
        for g in self._grasps.values():
            # Current world pose of robot grasp frame
            X_WR = self.plant.CalcRelativeTransform(self.plant_context, W, g.robot_frame)
            # Desired object world pose
            X_WO = X_WR.multiply(g.X_Ro)
            p_WO = X_WO.translation()

            # self.set_cube_xyz_fast("cube_1", p_WO)
            self.plant.SetPositions(self.plant_context, g.object_body.model_instance(), p_WO)
            # self.plant._set_cube_xyz_fast
            # self.plant.SetFreeBodyPose(self.plant_context, g.object_body, X_WO)

            if match_velocity:
                # Try to match object's spatial velocity to the robot's base body (good enough).
                # If your robot has a specific tool body, prefer that.
                try:
                    V_WR = self.plant.EvalBodySpatialVelocityInWorld(
                        self.plant_context, g.robot_frame.body()
                    )
                except Exception:
                    # Fallback: zero relative velocity
                    V_WR = SpatialVelocity.Zero()

                # set velocity
                self.plant.SetVelocities(self.plant_context, g.object_body.model_instance(), V_WR.translational())

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

    # @property
    # def nq(self) -> int:
    #     return self._nq

    # @property
    # def nv(self) -> int:
    #     return self._nv

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
        q = []
        qdot = []
        for name in self._controlled_names:
            mi = self.plant.GetModelInstanceByName(name)
            if "free_body" in name:
                body = self.plant.GetBodyByName("ee_link", mi)
                X_WE = self.plant.EvalBodyPoseInWorld(self.plant_context, body)
                q.append(X_WE.translation())
                q.append(X_WE.rotation().ToQuaternion().wxyz())

                Xdot_WE = self.plant.EvalBodySpatialVelocityInWorld(self.plant_context, body)
                qdot.append(Xdot_WE.translational())
                qdot.append(Xdot_WE.rotational())
            else:
                # TODO: check if this is correct for some robots, if necessary
                q.append(self.plant.GetPositions(self.plant_context, mi))
                qdot.append(self.plant.GetVelocities(self.plant_context, mi))

        for name in self._passive_names:
            mi = self.plant.GetModelInstanceByName(name)
            q.append(self.plant.GetPositions(self.plant_context, mi))
            qdot.append(self.plant.GetVelocities(self.plant_context, mi))

        return (np.concatenate(q), np.concatenate(qdot))

    @staticmethod
    def _compute_slices(lengths: List[int], start=0) -> List[slice]:
        """Return contiguous slices [0:n0), [n0:n0+n1), ... for given lengths."""
        s = start
        out = []
        for L in lengths:
            out.append(slice(s, s + L))
            s += L
        return out

    # def _apply_controlled_qv(self, action_q: np.ndarray, action_v: Optional[np.ndarray]):
    #     # For each controlled model, set q (and v) in its own coordinates.
    #     for i, mi in enumerate(self._controlled):
    #         q_slice = self._ctrl_q_slices[i]
    #         q_i = action_q[q_slice]
    #         self.plant.SetPositions(self.plant_context, mi, q_i)

    #         if action_v is not None:
    #             v_slice = self._ctrl_v_slices[i]
    #             v_i = action_v[v_slice]
    #         else:
    #             v_i = np.zeros(self._ctrl_nv[i])
    #         self.plant.SetVelocities(self.plant_context, mi, v_i)
