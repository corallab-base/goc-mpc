import numpy as np
import mujoco as mj
import meshcat
import meshcat.geometry as g


def _pose_matrix_from_mj(data, body_id: int):
    """4x4 homogeneous pose of a MuJoCo body in world frame."""
    T = np.eye(4)
    R = data.xmat[body_id].reshape(3, 3)  # row-major 3x3
    p = data.xpos[body_id]                # (3,)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


class MeshCatMirror:
    """Mirrors a subset of MuJoCo bodies into a MeshCat browser viewer."""
    def __init__(self, model: mj.MjModel, data: mj.MjData, bodies, radius=0.05):
        self.model, self.data = model, data
        self.vis = meshcat.Visualizer().open()  # opens browser tab
        # Build simple visuals (spheres) for the bodies you want to mirror.
        for name in bodies:
            path = self.vis[name]
            path.set_object(g.Sphere(radius), g.MeshLambertMaterial(color=0x1a66ff if "p1" in name else 0xff5522))
        # Optional ground
        self.vis["ground"].set_object(g.Box([10, 10, 0.02]), g.MeshLambertMaterial(color=0xdddddd))
        identity_matrix = meshcat.transformations.identity_matrix()
        self.vis["ground"].set_transform(identity_matrix)  # at z=0

        self.body_ids = {name: mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name) for name in bodies}

    def push(self):
        """Send current MuJoCo poses to MeshCat."""
        for name, bid in self.body_ids.items():
            self.vis[name].set_transform(_pose_matrix_from_mj(self.data, bid))
