"""Two-UR5e + Robotiq 2F-85 environment with differential IK.

Each arm is controlled independently by a damped-least-squares Jacobian
pseudo-inverse tracking its `attachment_site`.  A 6-DoF task-space velocity
is commanded at each substep: 3-D position tracking (primary) plus a full
SO(3) orientation task in the null-space that keeps each attachment-site
orientation identical to the configuration captured at reset().

The grippers are commanded open/close via the fingers_actuator; a "magic
grasp" keeps held cubes rigidly offset from the EE each substep so goc-mpc
sees correct cube poses.
"""

import numpy as np
import mujoco as mj
from pathlib import Path

try:
    import viser
    from mjviser.scene import ViserMujocoScene
    HAS_MJVISER = True
except Exception:
    HAS_MJVISER = False


_SCENE_XML = (
    Path(__file__).parent.parent / "descriptions" / "two_ur5e_2f85_scene.xml"
)

# UR5e home joint angles (standard pose)
_HOME_QPOS = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])

# Gripper ctrl: 0 = open, 255 = fully closed
_GRIPPER_OPEN   = 0.0
_GRIPPER_CLOSED = 200.0


class TwoUR5eDiffIKEnv:
    """Dual UR5e + Robotiq 2F-85 environment driven by differential IK.

    goc-mpc plans 3-D Cartesian trajectories for both end-effectors.
    step() tracks them using independent damped-LS IK for each arm.

    Observation returned by reset() / step():
        ((ee_pos_0, ee_vel_0), (ee_pos_1, ee_vel_1),
         (cube_pos_0, cube_vel_0), (cube_pos_1, cube_vel_1),
         (cube_pos_2, cube_vel_2))
    where each pos/vel has shape (3,).

    Args:
        model_path: path to MuJoCo scene XML (defaults to bundled scene).
        n_substeps:  physics substeps per control step.
        ik_kp:      proportional gain for EE position error (m/s per m).
        ik_ko:      proportional gain for orientation error (rad/s per rad).
        ik_lambda:  damping factor for damped-LS IK.
    """

    N_CUBES = 3

    def __init__(
        self,
        model_path: Path | str | None = None,
        n_substeps: int = 50,
        ik_kp: float = 50.0,
        ik_ko: float = 500.0,
        ik_lambda: float = 0.05,
    ):
        xml_path = str(model_path or _SCENE_XML)
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data  = mj.MjData(self.model)

        self.n_substeps = n_substeps
        self.kp  = ik_kp
        self.ko  = ik_ko
        self.lam = ik_lambda

        # --- site IDs for EE tracking ---
        self._site_id = [
            self._get_site("attachment_site_0"),
            self._get_site("attachment_site_1"),
        ]

        # --- joint qpos addresses for the two arms (6 each) ---
        arm0_names = [
            "ur5e_0_shoulder_pan_joint", "ur5e_0_shoulder_lift_joint",
            "ur5e_0_elbow_joint", "ur5e_0_wrist_1_joint",
            "ur5e_0_wrist_2_joint", "ur5e_0_wrist_3_joint",
        ]
        arm1_names = [
            "ur5e_1_shoulder_pan_joint", "ur5e_1_shoulder_lift_joint",
            "ur5e_1_elbow_joint", "ur5e_1_wrist_1_joint",
            "ur5e_1_wrist_2_joint", "ur5e_1_wrist_3_joint",
        ]
        self._arm_qpos_adr = [
            np.array([self.model.jnt_qposadr[self._jnt(n)] for n in arm0_names]),
            np.array([self.model.jnt_qposadr[self._jnt(n)] for n in arm1_names]),
        ]
        # qvel addresses mirror qpos addresses for revolute joints
        self._arm_qvel_adr = [
            np.array([self.model.jnt_dofadr[self._jnt(n)] for n in arm0_names]),
            np.array([self.model.jnt_dofadr[self._jnt(n)] for n in arm1_names]),
        ]

        # --- ctrl indices for arm joints and grippers ---
        act_names_0 = [
            "ur5e_0_shoulder_pan", "ur5e_0_shoulder_lift", "ur5e_0_elbow",
            "ur5e_0_wrist_1", "ur5e_0_wrist_2", "ur5e_0_wrist_3",
        ]
        act_names_1 = [
            "ur5e_1_shoulder_pan", "ur5e_1_shoulder_lift", "ur5e_1_elbow",
            "ur5e_1_wrist_1", "ur5e_1_wrist_2", "ur5e_1_wrist_3",
        ]
        self._arm_ctrl_idx = [
            np.array([self._act(n) for n in act_names_0]),
            np.array([self._act(n) for n in act_names_1]),
        ]
        self._gripper_ctrl_idx = [
            self._act("2f85_0_fingers_actuator"),
            self._act("2f85_1_fingers_actuator"),
        ]

        # --- cube body IDs and freejoint qpos addresses ---
        self._cube_body_id = [
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, f"cube_{i}")
            for i in range(self.N_CUBES)
        ]
        self._cube_jnt_qposadr = []
        self._cube_jnt_dofadr  = []
        for i in range(self.N_CUBES):
            jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, f"cube_{i}_joint")
            self._cube_jnt_qposadr.append(self.model.jnt_qposadr[jid])
            self._cube_jnt_dofadr.append(self.model.jnt_dofadr[jid])

        # gripper state: dict robot_id → (cube_id, world_offset) or None
        self._held: dict[int, tuple[int, np.ndarray] | None] = {0: None, 1: None}

        # Lookups used by apply_grasp_commands to convert goc-mpc name strings
        # ("ur5e_0_ee", "cube_1", …) to local integer indices.
        self._robot_name_to_idx: dict[str, int] = {
            "ur5e_0_ee": 0,
            "ur5e_1_ee": 1,
        }
        self._object_name_to_idx: dict[str, int] = {
            f"cube_{i}": i for i in range(self.N_CUBES)
        }

        # desired attachment-site rotation matrices, captured at reset()
        self._ee_R_des: list[np.ndarray] = [np.eye(3), np.eye(3)]

        self._viser_server = None
        self._viser_scene  = None

    # ------------------------------------------------------------------
    # Gym-style API
    # ------------------------------------------------------------------

    def reset(self):
        """Reset both arms to home configuration, cubes to initial poses."""
        mj.mj_resetData(self.model, self.data)
        # Set arm joint positions to home config
        for arm_adr in self._arm_qpos_adr:
            self.data.qpos[arm_adr] = _HOME_QPOS
        for arm_adr in self._arm_ctrl_idx:
            self.data.ctrl[arm_adr] = _HOME_QPOS
        # Grippers open
        self.data.ctrl[self._gripper_ctrl_idx[0]] = _GRIPPER_OPEN
        self.data.ctrl[self._gripper_ctrl_idx[1]] = _GRIPPER_OPEN
        self._held = {0: None, 1: None}
        mj.mj_forward(self.model, self.data)
        # Capture the start-configuration rotation matrix for each EE site
        for i, sid in enumerate(self._site_id):
            self._ee_R_des[i] = self.data.site_xmat[sid].reshape(3, 3).copy()
        if self._viser_scene is not None:
            self._viser_scene.update_from_mjdata(self.data)
        return self._get_obs(), {}

    def step(
        self,
        ee_target_pos_0: np.ndarray,
        ee_target_pos_1: np.ndarray,
        ee_target_vel_0: np.ndarray | None = None,
        ee_target_vel_1: np.ndarray | None = None,
    ):
        """Advance the simulation tracking both EE targets via diff IK.

        Args:
            ee_target_pos_0/1: desired EE position in world frame (3,).
            ee_target_vel_0/1: optional feedforward EE velocity (3,).

        Returns:
            obs, reward, terminated, truncated, info
        """
        sub_dt = self.model.opt.timestep
        lam2   = self.lam ** 2

        # Pre-allocate full-nv Jacobians for each arm (mj_jacSite fills them)
        Jp = [np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))]
        Jr = [np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))]

        targets = [
            (np.asarray(ee_target_pos_0, float), ee_target_vel_0),
            (np.asarray(ee_target_pos_1, float), ee_target_vel_1),
        ]

        for _ in range(self.n_substeps):
            # Magic grasp: teleport held cubes to maintain fixed EE offset
            for robot_id, held in self._held.items():
                if held is not None:
                    cube_id, offset = held
                    ee_pos = self.data.site_xpos[self._site_id[robot_id]].copy()
                    adr  = self._cube_jnt_qposadr[cube_id]
                    dadr = self._cube_jnt_dofadr[cube_id]
                    self.data.qpos[adr:adr+3]   = ee_pos + offset
                    self.data.qpos[adr+3:adr+7] = np.array([1.0, 0.0, 0.0, 0.0])
                    self.data.qvel[dadr:dadr+6] = 0.0

            # Hierarchical IK for each arm:
            #   Priority 1 (high): EE position tracking (kp)
            #   Priority 2 (low):  z-axis orientation aligned with [0,0,-1] (ko)
            #
            # Implemented as null-space projection so orientation correction only
            # uses the 3-D redundancy of the 6-DOF arm and cannot perturb position.
            # All updates are kinematic (direct qpos write) so the full commanded
            # joint velocity is applied every substep, bypassing actuator lag.
            # ctrl is synced to new_qpos so the servo holds against gravity.
            for i, (tgt_pos, tgt_vel) in enumerate(targets):
                sid     = self._site_id[i]
                dof_adr = self._arm_qvel_adr[i]

                mj.mj_jacSite(self.model, self.data, Jp[i], Jr[i], sid)
                # Work in the 6-DOF arm-joint subspace (ignore gripper/cube DOFs)
                Jp_arm = Jp[i][:, dof_adr]   # (3, 6)
                Jr_arm = Jr[i][:, dof_adr]   # (3, 6)

                # --- Priority 1: position tracking ---
                v_pos = self.kp * (tgt_pos - self.data.site_xpos[sid])
                if tgt_vel is not None:
                    v_pos = v_pos + np.asarray(tgt_vel, float)
                A1 = Jp_arm @ Jp_arm.T + lam2 * np.eye(3)
                Jp_pinv    = Jp_arm.T @ np.linalg.inv(A1)   # (6, 3)
                q_dot_pos  = Jp_pinv @ v_pos                 # (6,)

                # --- Priority 2: orientation in null-space of position ---
                # N projects joint velocities into the null-space of Jp_arm
                N = np.eye(6) - Jp_pinv @ Jp_arm             # (6, 6)
                # Full SO(3) error: world-frame angular velocity to drive R_site
                # toward the rotation captured at reset().  The skew-symmetric
                # part of R_des @ R_site.T encodes the rotation error for small
                # angles (valid since orientation drift is kept small every step).
                R_site = self.data.site_xmat[sid].reshape(3, 3)
                R_err  = self._ee_R_des[i] @ R_site.T
                v_rot  = self.ko * 0.5 * np.array([
                    R_err[2, 1] - R_err[1, 2],
                    R_err[0, 2] - R_err[2, 0],
                    R_err[1, 0] - R_err[0, 1],
                ])
                Jr_N   = Jr_arm @ N                          # (3, 6) effective
                A2 = Jr_N @ Jr_N.T + lam2 * np.eye(3)
                q_dot_rot = N @ Jr_N.T @ np.linalg.solve(A2, v_rot)  # (6,)

                q_dot_arm = q_dot_pos + q_dot_rot

                # Kinematic update; sync ctrl so the servo holds configuration.
                qpos_adr = self._arm_qpos_adr[i]
                new_qpos = self.data.qpos[qpos_adr] + q_dot_arm * sub_dt
                self.data.qpos[qpos_adr] = new_qpos
                self.data.ctrl[self._arm_ctrl_idx[i]] = new_qpos
                mj.mj_kinematics(self.model, self.data)

            mj.mj_step(self.model, self.data)

        if self._viser_scene is not None:
            self._viser_scene.update_from_mjdata(self.data)

        return self._get_obs(), 0.0, False, False, {}

    def apply_grasp_commands(self, grasp_commands: list):
        """Execute grasp commands from goc-mpc.

        Each command is a tuple ``(cmd, robot_name, object_name)`` where
        cmd is ``"grab"`` or ``"release"``, robot_name is e.g. ``"ur5e_0_ee"``,
        and object_name is e.g. ``"cube_0"``.  "grab" closes the gripper and
        records a world-frame EE→cube offset so the cube rides with the EE via
        the magic-grasp in step().
        """
        for cmd, robot_name, object_name in grasp_commands:
            robot_id  = self._robot_name_to_idx[robot_name]
            object_id = self._object_name_to_idx[object_name]
            if cmd == "grab":
                ee_pos   = self.data.site_xpos[self._site_id[robot_id]].copy()
                cube_pos = self.data.xpos[self._cube_body_id[object_id]].copy()
                offset   = cube_pos - ee_pos
                self._held[robot_id] = (object_id, offset)
                self.data.ctrl[self._gripper_ctrl_idx[robot_id]] = _GRIPPER_CLOSED
            elif cmd == "release":
                self._held[robot_id] = None
                self.data.ctrl[self._gripper_ctrl_idx[robot_id]] = _GRIPPER_OPEN

    # ------------------------------------------------------------------
    # Viewer
    # ------------------------------------------------------------------

    def launch_viewer(self, port: int = 7000):
        if not HAS_MJVISER:
            raise RuntimeError("mjviser not available. pip install mjviser")
        self._viser_server = viser.ViserServer(port=port)
        self._viser_scene  = ViserMujocoScene(self._viser_server, self.model, num_envs=1)
        self._viser_scene.create_visualization_gui()
        self._viser_scene.update_from_mjdata(self.data)
        print(f"mjviser running at http://localhost:{port}")

    def close_viewer(self):
        self._viser_scene = None
        if self._viser_server is not None:
            self._viser_server.stop()
            self._viser_server = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self):
        obs = []
        for sid in self._site_id:
            ee_pos = self.data.site_xpos[sid].copy()
            J_pos  = np.zeros((3, self.model.nv))
            mj.mj_jacSite(self.model, self.data, J_pos, None, sid)
            ee_vel = J_pos @ self.data.qvel
            obs.append((ee_pos, ee_vel))

        for body_id in self._cube_body_id:
            cube_pos = self.data.xpos[body_id].copy()
            cube_vel = self.data.cvel[body_id][3:6].copy()  # linear part of 6D vel
            obs.append((cube_pos, cube_vel))

        return tuple(obs)

    def _get_site(self, name: str) -> int:
        sid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise RuntimeError(f"Site '{name}' not found in model")
        return sid

    def _jnt(self, name: str) -> int:
        jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"Joint '{name}' not found in model")
        return jid

    def _act(self, name: str) -> int:
        aid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise RuntimeError(f"Actuator '{name}' not found in model")
        return aid
