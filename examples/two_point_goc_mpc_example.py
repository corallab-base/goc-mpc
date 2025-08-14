import os
import imageio
import numpy as np
import mujoco as mj
import matplotlib.pyplot as plt

from mujoco import viewer

from goc_mpc.systems.two_points import TwoPointMassEnv
from goc_mpc.goc_mpc import GraphOfConstraintsMPC
from goc_mpc.utils.mesh_cat_mirror import MeshCatMirror


def record_video(
    env,
    action_fn,
    T,
    out_path="out.mp4",
    fps=50,
    width=640,
    height=480,
    camera=None,
    start_qpos=None,
    start_qvel=None,
):
    """
    Render a video by stepping the env with action_fn and writing frames to disk.

    Args:
      env: TwoPointMassEnv instance (or any mujoco env exposing .model, .data, .step()).
      action_fn: callable(obs, k) -> action (np.ndarray of shape env.action_dim).
      T: number of steps to record.
      out_path: output video file path (e.g., 'demo.mp4').
      fps: output video frame rate.
      width, height: pixel size.
      camera: None or string camera name defined in the model. None uses the free camera.
      start_qpos, start_qvel: optional initial state to reset to.

    Notes:
      - For headless Linux, set: export MUJOCO_GL=egl  (before Python starts).
      - For odd widths, we disable macroblock alignment in imageio to avoid artifacts.
    """
    # If you're on a headless server, uncomment this (must be set before importing mujoco):
    # os.environ.setdefault("MUJOCO_GL", "egl")

    # Reset environment to a known state.
    obs, _ = env.reset(qpos=start_qpos, qvel=start_qvel)

    # Create an offscreen renderer once.
    renderer = mj.Renderer(env.model, height=height, width=width)

    # Open an ffmpeg writer (H.264).
    writer = imageio.get_writer(
        out_path, fps=fps, codec="libx264", quality=8, macro_block_size=None
    )

    try:
        for k in range(T):
            action = action_fn(obs, k)
            obs, reward, terminated, truncated, info = env.step(action)

            # Update the scene and render an RGB frame (uint8 HxWx3).
            renderer.update_scene(env.data, camera=camera)
            frame = renderer.render()

            writer.append_data(frame)

            if terminated or truncated:
                break
    finally:
        writer.close()


def record_video_from_qpos(
    env,
    qpos_traj,
    out_path="traj.mp4",
    fps=50,
    width=640,
    height=480,
    camera=None,
):
    """
    Convenience: record a video from a *precomputed configuration trajectory*.
    Assumes env.mode == 'teleport' (direct configuration setting each step).

    Args:
      qpos_traj: array of shape [T, env.model.nq], each row is a full qpos.
    """
    qpos_traj = np.asarray(qpos_traj)
    assert qpos_traj.ndim == 2 and qpos_traj.shape[1] == env.model.nq, \
        f"Expected shape [T, {env.model.nq}], got {qpos_traj.shape}"

    def action_fn(obs, k):
        return qpos_traj[k]

    record_video(
        env,
        action_fn=action_fn,
        T=qpos_traj.shape[0],
        out_path=out_path,
        fps=fps,
        width=width,
        height=height,
        camera=camera,
        start_qpos=qpos_traj[0],
        start_qvel=np.zeros(env.model.nv),
    )


def two_point_example():
    # env = TwoPointMassEnv(mode="servo", n_substeps=5)

    # T = 600
    # t = np.linspace(0.0, 12.0, T)
    # p1 = np.stack([0.8*np.cos(0.6*t), 0.8*np.sin(0.6*t), 0.9 + 0.1*np.sin(1.2*t)], axis=1)
    # p2 = np.stack([0.6*np.sin(0.9*t+0.5), 0.4*np.sin(0.6*t), 0.8 + 0.1*np.cos(1.0*t)], axis=1)
    # qpos_traj = np.concatenate([p1, p2], axis=1)  # [T, 6]


    # def policy(obs, k):
    #     return qpos_traj[k]  # desired joint positions

    # record_video(env, policy, T=len(qpos_traj), out_path="two_points_servo.mp4", fps=50, camera="topdown")

    env = TwoPointMassEnv(mode="servo", n_substeps=5)
    obs, _ = env.reset(qpos=np.array([0.0, 0.0, 1.0,   0.5, 0.0, 0.8]))

    mirror = MeshCatMirror(env.model, env.data, bodies=["p1", "p2"], radius=0.05)

    # GoC-MPC
    mpc = GraphOfConstraintsMPC()

    # Demo: two simple Lissajous-ish trajectories played back
    T = 600
    t = np.linspace(0.0, 12.0, T)
    p1 = np.stack([0.8*np.cos(0.6*t), 0.8*np.sin(0.6*t), 0.9 + 0.1*np.sin(1.2*t)], axis=1)
    p2 = np.stack([0.6*np.sin(0.9*t+0.5), 0.4*np.sin(0.6*t), 0.8 + 0.1*np.cos(1.0*t)], axis=1)

    observed_qs = []

    for k in range(T):
        qpos = np.concatenate([p1[k], p2[k]], axis=0)
        obs, rew, done, trunc, info = env.step(qpos)

        observed_qs.append(obs[:6])
        
        # If you want to slow it down to (roughly) real-time:
        import time; time.sleep(0.01)

        mirror.push()            # …and push poses to the browser viewer

    ref_qs = np.concatenate((p1, p2), axis=-1)
    observed_qs = np.stack(observed_qs)
    error = observed_qs - ref_qs

    print(error)

    breakpoint()

    



    # state = env.reset()

    # for pos in positions:
    #     start_x = np.array([0.1, -1.0 + pos])
    #     start_v = np.array([0.0, 2.0])

    #     mpc.solve(start_x, start_v, 1);
    #     mpc.fill_cubic_spline(sp, start_x, start_v)

    #     t_vals = np.linspace(sp.begin(), sp.end(), 100)
    #     positions = sp.eval_multiple(t_vals, 0)
    #     ax.plot(positions[:, 0], positions[:, 1], label=f'Path for {pos}')

    # fig.savefig("./test.png")


if __name__ == "__main__":
    two_point_example()
