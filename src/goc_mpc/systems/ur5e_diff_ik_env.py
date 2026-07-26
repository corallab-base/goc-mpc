import numpy as np
import mujoco as mj
from pathlib import Path

try:
    import viser
    from mjviser.scene import ViserMujocoScene
    HAS_MJVISER = True
except Exception:
    HAS_MJVISER = False


# Default: MuJoCo Menagerie UR5e (installed via robot_descriptions or manually)
_MENAGERIE_SCENE = (
    Path.home() / ".cache" / "robot_descriptions" /
    "mujoco_menagerie" / "universal_robots_ur5e" / "scene.xml"
)


class UR5eDiffIKEnv:
    """MuJoCo UR5e environment driven by differential IK.

    goc-mpc plans a 3-D Cartesian trajectory for the end-effector.
    step() tracks that trajectory by computing a damped-least-squares
    Jacobian pseudo-inverse at each physics substep.

    Observation returned by reset() and step() is a tuple
        (ee_pos, ee_vel)  with shapes (3,) each,
    matching the SimpleDrakeGym convention so it slots directly into the
    x, x_dot = obs unpacking used in the pointmass_example pattern.

    Args:
        model_path: path to the MuJoCo scene XML (must resolve mesh assets).
                    Defaults to the MuJoCo Menagerie UR5e scene.
        n_substeps:  physics substeps per control call (timestep × n_substeps
                    = wall time advanced per step()).
        ik_kp:      proportional gain for the end-effector position error.
        ik_lambda:  damping factor for the damped-least-squares IK.
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        n_substeps: int = 50,
        ik_kp: float = 10.0,
        ik_lambda: float = 0.05,
    ):
        xml_path = str(model_path or _MENAGERIE_SCENE)
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)

        self.n_substeps = n_substeps
        self.kp = ik_kp
        self.lam = ik_lambda

        self._site_id = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_SITE, "attachment_site"
        )
        if self._site_id < 0:
            raise RuntimeError(
                "attachment_site not found in UR5e model. "
                "Check that the correct scene XML is being loaded."
            )

        # keyframe index for the home configuration
        self._home_key = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_KEY, "home"
        )

        self._viser_server = None
        self._viser_scene = None

    # ------------------------------------------------------------------
    # Gym-style API
    # ------------------------------------------------------------------

    def reset(self):
        """Reset to the home keyframe. Returns (ee_pos, ee_vel), info."""
        mj.mj_resetData(self.model, self.data)
        if self._home_key >= 0:
            mj.mj_resetDataKeyframe(self.model, self.data, self._home_key)
        mj.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, ee_target_pos: np.ndarray, ee_target_vel: np.ndarray | None = None):
        """Advance the simulation while tracking ee_target_pos via diff IK.

        At each physics substep the controller computes:
            v_des = kp * (target - current_ee) [+ feedforward]
            q_dot = J^T (J J^T + λ² I)^{-1} v_des   (damped LS)
            ctrl  = q + q_dot * dt

        Args:
            ee_target_pos: desired end-effector position in world frame (3,).
            ee_target_vel: optional feedforward EE velocity (3,).

        Returns:
            (ee_pos, ee_vel), reward, terminated, truncated, info
        """
        ee_target_pos = np.asarray(ee_target_pos, dtype=float)
        sub_dt = self.model.opt.timestep

        J_pos = np.zeros((3, self.model.nv))
        J_rot = np.zeros((3, self.model.nv))

        for _ in range(self.n_substeps):
            mj.mj_jacSite(self.model, self.data, J_pos, J_rot, self._site_id)

            ee_pos = self.data.site_xpos[self._site_id]
            vel_des = self.kp * (ee_target_pos - ee_pos)
            if ee_target_vel is not None:
                vel_des = vel_des + ee_target_vel

            # Damped least-squares pseudo-inverse (position only)
            A = J_pos @ J_pos.T + self.lam ** 2 * np.eye(3)
            q_dot = J_pos.T @ np.linalg.solve(A, vel_des)

            self.data.ctrl[:] = self.data.qpos + q_dot * sub_dt
            mj.mj_step(self.model, self.data)

        if self._viser_scene is not None:
            self._viser_scene.update_from_mjdata(self.data)

        return self._get_obs(), 0.0, False, False, {}

    # ------------------------------------------------------------------
    # Viewer helpers
    # ------------------------------------------------------------------

    def launch_viewer(self, port: int = 8080):
        """Start an mjviser web server and open the UR5e scene.

        Connect from any browser at  http://<host>:<port>
        step() pushes updated poses after every control call.
        """
        if not HAS_MJVISER:
            raise RuntimeError("mjviser / viser not available. Install with: pip install mjviser")
        self._viser_server = viser.ViserServer(port=port)
        self._viser_scene = ViserMujocoScene(self._viser_server, self.model, num_envs=1)
        self._viser_scene.create_visualization_gui()
        self._viser_scene.update_from_mjdata(self.data)
        print(f"mjviser running at http://localhost:{port}")

    def close_viewer(self):
        self._viser_scene = None
        if self._viser_server is not None:
            self._viser_server.stop()
            self._viser_server = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def get_ee_pos(self) -> np.ndarray:
        """Current end-effector position in world frame."""
        return self.data.site_xpos[self._site_id].copy()

    def _get_obs(self):
        ee_pos = self.data.site_xpos[self._site_id].copy()
        J_pos = np.zeros((3, self.model.nv))
        mj.mj_jacSite(self.model, self.data, J_pos, None, self._site_id)
        ee_vel = J_pos @ self.data.qvel
        return ee_pos, ee_vel
