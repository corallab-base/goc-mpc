# pip install mujoco gymnasium
import numpy as np
import mujoco as mj

try:
    # Optional viewer; if unavailable, everything else still works headless.
    from mujoco import viewer
    HAS_VIEWER = True
except Exception:
    HAS_VIEWER = False

from goc_mpc.utils.mesh_cat_mirror import MeshCatMirror


MJCF_TWO_POINT_MASSES = r"""
<mujoco model="two_point_masses">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <compiler angle="radian"/>
  <worldbody>
    <light name="sun"
           directional="true"
           pos="2 -2 3"
           dir="-2 2 -3"
           diffuse="1 1 1"
           specular="0.2 0.2 0.2"/>

    <!-- A softer fill light from the opposite side (optional, helps avoid harsh contrast) -->
    <light name="fill"
           directional="true"
           pos="-2 2 2"
           dir="2 -2 -2"
           diffuse="0.4 0.4 0.4"
           specular="0 0 0"/>

    <camera name="topdown" pos="0 0 3.0" zaxis="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.1" rgba="0.8 0.9 1 1"/>
    <body name="p1" pos="0 0 1.0">
      <!-- Three orthogonal sliders => qpos: x1,y1,z1 -->
      <joint name="p1_x" type="slide" axis="1 0 0" damping="0.1"/>
      <joint name="p1_y" type="slide" axis="0 1 0" damping="0.1"/>
      <joint name="p1_z" type="slide" axis="0 0 1" damping="0.1"/>
      <geom type="sphere" size="0.05" rgba="0.1 0.4 1 1" mass="1.0"/>
    </body>
    <body name="p2" pos="0.5 0.0 0.8">
      <!-- qpos: x2,y2,z2 -->
      <joint name="p2_x" type="slide" axis="1 0 0" damping="0.1"/>
      <joint name="p2_y" type="slide" axis="0 1 0" damping="0.1"/>
      <joint name="p2_z" type="slide" axis="0 0 1" damping="0.1"/>
      <geom type="sphere" size="0.05" rgba="1 0.3 0.1 1" mass="1.0"/>
    </body>
  </worldbody>

  <!-- Actuators: position servos on each slider (used in mode='servo') -->
  <actuator>
    <position name="p1_x_m" joint="p1_x" kp="200" kv="20"/>
    <position name="p1_y_m" joint="p1_y" kp="200" kv="20"/>
    <position name="p1_z_m" joint="p1_z" kp="300" kv="30"/>
    <position name="p2_x_m" joint="p2_x" kp="200" kv="20"/>
    <position name="p2_y_m" joint="p2_y" kp="200" kv="20"/>
    <position name="p2_z_m" joint="p2_z" kp="300" kv="30"/>
  </actuator>
</mujoco>
"""


class TwoPointMassEnv:
    """
    A minimal, gym-like MuJoCo environment for two point masses.

    Modes:
      - 'teleport': step(action) sets qpos[:] = action and calls forward/step.
      - 'servo':    step(action) sets ctrl[:] = action (desired positions) and steps physics.

    Observation = np.concatenate([qpos(6), qvel(6)]) -> shape (12,)
    Action = positions for both bodies -> shape (6,), [x1,y1,z1,x2,y2,z2]
    """
    metadata = {"render_modes": ["human", "none"]}

    def __init__(self, mode="teleport", n_substeps=10, xml_str=MJCF_TWO_POINT_MASSES):
        assert mode in ("teleport", "servo")
        self.mode = mode
        self.n_substeps = int(n_substeps)

        self.model = mj.MjModel.from_xml_string(xml_str)
        self.data = mj.MjData(self.model)

        # Indices: qpos = [x1,y1,z1, x2,y2,z2], ctrl = same ordering via position actuators
        self.action_dim = 6
        self.obs_dim = self.model.nq + self.model.nv  # 6 + 6 = 12

    def reset(self, qpos=None, qvel=None):
        mj.mj_resetData(self.model, self.data)
        if qpos is not None:
            assert qpos.shape == (self.model.nq,)
            self.data.qpos[:] = qpos
        if qvel is not None:
            assert qvel.shape == (self.model.nv,)
            self.data.qvel[:] = qvel
        mj.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=float)
        assert action.shape == (self.action_dim,)

        if self.mode == "teleport":
            # Directly set configuration for this step (state player).
            self.data.qpos[:] = action
            # Optionally damp velocities to keep things tame
            self.data.qvel[:] = 0.0
            mj.mj_forward(self.model, self.data)
            # A small number of integration steps to resolve contacts/constraints/viscosity.
            for _ in range(self.n_substeps):
                mj.mj_step(self.model, self.data)

        else:  # 'servo'
            # Position actuators take desired joint positions in ctrl (same ordering).
            self.data.ctrl[:] = action
            for _ in range(self.n_substeps):
                mj.mj_step(self.model, self.data)

        obs = self._get_obs()
        # No task reward/termination here; you can add your own logic if needed.
        reward, terminated, truncated, info = 0.0, False, False, {}
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()], axis=0)

    def render(self, render_mode="human"):
        if render_mode != "human":
            return
        if not HAS_VIEWER:
            raise RuntimeError("mujoco.viewer not available in this environment.")
        # Simple one-off viewer that shows the current state; call from a loop while stepping.
        viewer.launch_passive(self.model, self.data)

    # Convenience helpers
    def set_qpos(self, qpos):
        assert qpos.shape == (self.model.nq,)
        self.data.qpos[:] = qpos
        mj.mj_forward(self.model, self.data)

    def set_qvel(self, qvel):
        assert qvel.shape == (self.model.nv,)
        self.data.qvel[:] = qvel


if __name__ == "__main__":
    env = TwoPointMassEnv(mode="teleport", n_substeps=5)
    obs, _ = env.reset(qpos=np.array([0.0, 0.0, 1.0,   0.5, 0.0, 0.8]))

    mirror = MeshCatMirror(env.model, env.data, bodies=["p1", "p2"], radius=0.05)

    # Demo: two simple Lissajous-ish trajectories played back
    T = 600
    t = np.linspace(0.0, 12.0, T)
    p1 = np.stack([0.8*np.cos(0.6*t), 0.8*np.sin(0.6*t), 0.9 + 0.1*np.sin(1.2*t)], axis=1)
    p2 = np.stack([0.6*np.sin(0.9*t+0.5), 0.4*np.sin(0.6*t), 0.8 + 0.1*np.cos(1.0*t)], axis=1)

    # If you want to see the viewer, open a terminal window to watch and keep the loop slow.
    use_viewer = False and HAS_VIEWER
    if use_viewer:
        # Launch an interactive viewer in another window
        viewer.launch_passive(env.model, env.data)

    for k in range(T):
        qpos = np.concatenate([p1[k], p2[k]], axis=0)
        obs, rew, done, trunc, info = env.step(qpos)

        # If you want to slow it down to (roughly) real-time:
        import time; time.sleep(0.02)

        mirror.push()            # …and push poses to the browser viewer

    # # Now try 'servo' mode (physics moves masses to your targets)
    # env = TwoPointMassEnv(mode="servo", n_substeps=10)
    # env.reset()
    # for k in range(T):
    #     qpos_des = np.concatenate([p1[k], p2[k]], axis=0)   # desired positions
    #     obs, *_ = env.step(qpos_des)

        # mirror.push()            # …and push poses to the browser viewer
