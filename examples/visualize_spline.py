import numpy as np
import matplotlib.pyplot as plt

from goc_mpc.splines import CubicSpline


times = np.array([0.0, 1.0, 2.0, 3.0])

# === Define interesting control points ===
pts1 = np.array([[0.0, 0.0],
                [1.0, 2.0],
                [3.0, 1.0],
                [4.0, 3.0]])

vels1 = np.array([[0.0, 0.0],
                 [1.0, 0.0],
                 [0.0, 1.0],
                 [0.0, 0.0]])

# === Construct spline ===
spline1 = CubicSpline()
spline1.set(pts1, vels1, times)


# === Define interesting control points ===
pts2 = np.array([[0.0, 0.5],
                 [1.0, 4.0],
                 [2.0, 0.6],
                 [4.0, 0.0]])

vels2 = np.array([[0.0, 0.0],
                  [0.5, 0.5],
                  [1.5, 0.0],
                  [0.0, 0.0]])

# === Construct spline ===
spline2 = CubicSpline()
spline2.set(pts2, vels2, times)

# === Plotting ===
fig, axes = plt.subplots(2, 1, figsize=(8, 12))

for i, spline, pts, vels in zip(range(2),
                                [spline1, spline2],
                                [pts1, pts2],
                                [vels1, vels2]):
    # === Evaluate over dense time grid ===
    t_vals = np.linspace(times[0], times[-1], 300)
    positions = spline.eval_multiple(t_vals, 0)
    velocities = spline.eval_multiple(t_vals, 1)
    # accelerations = spline.eval_multiple(t_vals, 2)
    
    
    # Plot path
    axes[0].plot(positions[:, 0], positions[:, 1], label=f'Spline {i} Path', color='blue')
    axes[0].plot(pts[:, 0], pts[:, 1], 'ro', label='Waypoints')
    axes[0].quiver(pts[:, 0], pts[:, 1], vels[:, 0], vels[:, 1], angles='xy', scale_units='xy', scale=1, color='green', label='Velocities')
    axes[0].set_title(f'Spline {i} Path')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].axis('equal')
    axes[0].legend()
    
    # Plot speed (velocity magnitude) over time
    speed = np.linalg.norm(velocities, axis=1)
    axes[1].plot(t_vals, speed, label=f'Speed {i}', color='orange')
    axes[1].set_title('Speed vs Time')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Speed')
    axes[1].legend()
    
    # # Plot velocity over time
    # axes[1].plot(t_vals, velocities[:, 0], label='vx', color='orange')
    # axes[1].plot(t_vals, velocities[:, 1], label='vy', color='red')
    # axes[1].set_title('Velocity vs Time')
    # axes[1].set_xlabel('Time')
    # axes[1].set_ylabel('Velocity')
    # axes[1].legend()
    
    # Plot acceleration over time
    # axes[2].plot(t_vals, accelerations[:, 0], label='ax', color='purple')
    # axes[2].plot(t_vals, accelerations[:, 1], label='ay', color='magenta')
    # axes[2].set_title('Acceleration vs Time')
    # axes[2].set_xlabel('Time')
    # axes[2].set_ylabel('Acceleration')
    # axes[2].legend()
    
plt.tight_layout()
plt.show()
