import numpy as np
import plotly.graph_objects as go

# --------------------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------------------
def _quat_to_rotmat(q, order="wxyz"):
    q = np.asarray(q, dtype=float)
    if order == "wxyz":
        w, x, y, z = q
    elif order == "xyzw":
        x, y, z, w = q
    else:
        raise ValueError("quat_order must be 'wxyz' or 'xyzw'")
    n = np.linalg.norm([w, x, y, z])
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x*x + y*y)]
    ])

def _split_pos_quat(q, quat_order="wxyz"):
    q = np.asarray(q, dtype=float).ravel()
    pos = q[:3]
    quat = q[3:7]
    quat = quat / np.linalg.norm(quat)
    return pos, quat

def _get_q_from_eval(eval_out):
    # Accept q or (q, v, a) or (q, v)
    if isinstance(eval_out, (list, tuple)):
        if len(eval_out) >= 1:
            return np.asarray(eval_out[0])
    return np.asarray(eval_out)

def _seg(p, v, L):
    p = np.asarray(p); v = np.asarray(v)
    return [p[0], p[0] + L*v[0], None], [p[1], p[1] + L*v[1], None], [p[2], p[2] + L*v[2], None]

# --------------------------------------------------------------------------------------
# Main function
# --------------------------------------------------------------------------------------
def render_spline_to_html(
    spline,
    html_out="spline_viewer.html",
    *,
    quat_order="wxyz",
    n_frames=200,
    axis_length=0.15,
    path_samples=400,
    t_min=None,
    t_max=None,
    title="R³ × SO(3) Spline Viewer",
    include_knots=None  # optional: list/array of knot positions to show as markers
):
    """
    Render an interactive 3D viewer (Plotly) for a CubicConfigurationSpline on R^3 x SO(3).

    Parameters
    ----------
    spline : CubicConfigurationSpline
        Your spline instance. Its eval(t) should return q or (q, v, a) where
        q = [x, y, z, quat(4)] with quat_order specified by `quat_order`.
    html_out : str
        Output HTML path (self-contained; no internet needed).
    quat_order : {"wxyz", "xyzw"}
        Quaternion storage order in q.
    n_frames : int
        Number of animation frames between t_min and t_max.
    axis_length : float
        Length of the pose triad axes.
    path_samples : int
        Number of samples to draw the static position path.
    t_min, t_max : float or None
        If None, tries spline.begin()/spline.end(). Otherwise uses provided range.
    title : str
        Plot title.
    include_knots : np.ndarray or list or None
        Optional Nx3 array of control point positions to plot as markers.

    Returns
    -------
    str
        The html_out path.
    """
    # Determine time bounds
    if t_min is None:
        t_min = spline.begin() if hasattr(spline, "begin") else 0.0
    if t_max is None:
        t_max = spline.end() if hasattr(spline, "end") else 1.0
    if t_max <= t_min:
        raise ValueError("t_max must be > t_min.")

    # Sample static path (positions only) for context
    t_dense = np.linspace(t_min, t_max, path_samples)
    path_pos = []
    for t in t_dense:
        q_full = _get_q_from_eval(spline.eval(t))
        pos, _ = _split_pos_quat(q_full, quat_order)
        path_pos.append(pos)
    path_pos = np.array(path_pos)

    # Compute bounds
    pts = [path_pos]
    if include_knots is not None:
        include_knots = np.asarray(include_knots, dtype=float)
        if include_knots.ndim == 2 and include_knots.shape[1] == 3:
            pts.append(include_knots)
    all_xyz = np.vstack(pts)
    mins = all_xyz.min(axis=0) - 0.1
    maxs = all_xyz.max(axis=0) + 0.1

    # Animation frames
    t_frames = np.linspace(t_min, t_max, n_frames)
    frames = []
    for t in t_frames:
        q_full = _get_q_from_eval(spline.eval(t))
        pos, quat = _split_pos_quat(q_full, quat_order)
        R = _quat_to_rotmat(quat, quat_order)
        ex, ey, ez = R[:, 0], R[:, 1], R[:, 2]
        px, py, pz = pos

        # Triad segments
        x_x, x_y, x_z = _seg(pos, ex, axis_length)
        y_x, y_y, y_z = _seg(pos, ey, axis_length)
        z_x, z_y, z_z = _seg(pos, ez, axis_length)

        frames.append(go.Frame(
            name=f"{t:.6f}",
            data=[
                go.Scatter3d(x=[px], y=[py], z=[pz], mode="markers"),           # moving point
                go.Scatter3d(x=x_x, y=x_y, z=x_z, mode="lines"),                # x-axis
                go.Scatter3d(x=y_x, y=y_y, z=y_z, mode="lines"),                # y-axis
                go.Scatter3d(x=z_x, y=z_y, z=z_z, mode="lines"),                # z-axis
            ]
        ))

    # Static traces
    path_trace = go.Scatter3d(
        x=path_pos[:, 0], y=path_pos[:, 1], z=path_pos[:, 2],
        mode="lines", name="path", line=dict(width=4)
    )
    traces = [path_trace]

    if include_knots is not None and include_knots.size > 0:
        knot_trace = go.Scatter3d(
            x=include_knots[:, 0], y=include_knots[:, 1], z=include_knots[:, 2],
            mode="markers", name="knots", marker=dict(size=4)
        )
        traces.append(knot_trace)

    # Initial pose (first frame)
    init_q = _get_q_from_eval(spline.eval(t_frames[0]))
    init_pos, init_quat = _split_pos_quat(init_q, quat_order)
    R0 = _quat_to_rotmat(init_quat, quat_order)
    ex0, ey0, ez0 = R0[:, 0], R0[:, 1], R0[:, 2]
    px0, py0, pz0 = init_pos
    x_x, x_y, x_z = _seg(init_pos, ex0, axis_length)
    y_x, y_y, y_z = _seg(init_pos, ey0, axis_length)
    z_x, z_y, z_z = _seg(init_pos, ez0, axis_length)

    moving_point = go.Scatter3d(x=[px0], y=[py0], z=[pz0], mode="markers", name="pose")
    axis_x = go.Scatter3d(x=x_x, y=x_y, z=x_z, mode="lines", name="x-axis")
    axis_y = go.Scatter3d(x=y_x, y=y_y, z=y_z, mode="lines", name="y-axis")
    axis_z = go.Scatter3d(x=z_x, y=z_y, z=z_z, mode="lines", name="z-axis")

    traces.extend([moving_point, axis_x, axis_y, axis_z])

    # Slider with ~10 labeled steps (the animation itself uses n_frames)
    slider_steps = [
        dict(method="animate",
             args=[[f"{tt:.6f}"],
                   dict(mode="immediate",
                        frame=dict(duration=0, redraw=True),
                        transition=dict(duration=0))],
             label=f"{tt:.2f}")
        for tt in np.linspace(t_min, t_max, 10)
    ]
    sliders = [dict(
        active=0,
        pad={"t": 30},
        currentvalue={"prefix": "t = "},
        steps=slider_steps
    )]

    updatemenus = [dict(
        type="buttons",
        showactive=False,
        y=1.08, x=0.0, xanchor="left",
        buttons=[
            dict(label="Play",
                 method="animate",
                 args=[None, dict(frame=dict(duration=max(1, int(1000/max(n_frames,1)))),  # ~1s per 60 frames
                                  transition=dict(duration=0),
                                  fromcurrent=True,
                                  mode="immediate")]),
            dict(label="Pause",
                 method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False),
                                    transition=dict(duration=0),
                                    mode="immediate")]),
        ]
    )]

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            xaxis=dict(range=[mins[0], maxs[0]]),
            yaxis=dict(range=[mins[1], maxs[1]]),
            zaxis=dict(range=[mins[2], maxs[2]]),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        updatemenus=updatemenus,
        sliders=sliders,
    )

    fig = go.Figure(data=traces, layout=layout, frames=frames)
    # Self-contained: no internet required to open the HTML
    fig.write_html(html_out, include_plotlyjs=True, auto_play=False)
    return html_out

# --------------------------------------------------------------------------------------
# Example usage (you can remove below here if you already have your own spline)
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    from goc_mpc.splines import CubicConfigurationSpline, Block

    # Times
    times = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)

    # A small, interesting R^3 path and changing SO(3) orientation
    def axis_angle_to_quat(axis, angle_rad, order="wxyz"):
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        s = np.sin(angle_rad / 2.0)
        w = np.cos(angle_rad / 2.0)
        x, y, z = axis * s
        return np.array([w, x, y, z]) if order == "wxyz" else np.array([x, y, z, w])

    p0 = np.array([0.00, 0.00, 0.00])
    p1 = np.array([0.60, 0.30, 0.20])
    p2 = np.array([0.20, 0.80, 0.50])
    p3 = np.array([0.85, 0.20, 0.10])

    q0 = axis_angle_to_quat([0, 0, 1], np.deg2rad(0))
    q1 = axis_angle_to_quat([1, 1, 0], np.deg2rad(120))
    q2 = axis_angle_to_quat([0, 1, 1], np.deg2rad(240))
    q3 = axis_angle_to_quat([1, 0, 1], np.deg2rad(360))

    pts1 = np.vstack([
        np.hstack([p0, q0]),
        np.hstack([p1, q1]),
        np.hstack([p2, q2]),
        np.hstack([p3, q3]),
    ])

    lin_v = np.array([
        [0.9,  0.5,  0.3],
        [-0.2, 0.7,  0.4],
        [0.8, -0.5,  0.0],
        [0.1, -0.6, -0.2],
    ])
    ang_v = np.array([
        [0.0,   0.0,   1.2],
        [0.9,   0.9,   0.0],
        [0.0,   1.0,   1.0],
        [0.8,   0.0,   0.8],
    ])
    vels1 = np.hstack([lin_v, ang_v])

    spline_spec = [Block.R(3), Block.SO3()]
    spline1 = CubicConfigurationSpline(spline_spec)
    spline1.set(pts1, vels1, times)

    html_path = render_spline_to_html(
        spline1,
        html_out="spline_viewer.html",
        quat_order="wxyz",
        n_frames=240,
        axis_length=0.15,
        path_samples=500,
        title="R³ × SO(3) Spline (Demo)",
        include_knots=pts1[:, :3],
    )
    print(f"Wrote {html_path}")
    print("Serve it (e.g.) with:  python -m http.server 8000")
