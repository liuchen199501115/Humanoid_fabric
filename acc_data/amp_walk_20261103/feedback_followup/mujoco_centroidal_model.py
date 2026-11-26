"""Fast offline centroidal model; uses the same URDF, no simulation stepping."""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from mujoco_reference_dynamics import ReferenceDynamics

class CentroidalModel(ReferenceDynamics):
    def __init__(self, urdf, names):
        super().__init__(urdf, names)
        self.mass = float(self.model.body_mass.sum())

    def _state(self, q, v=None):
        m, d = self.model, self.data
        rotation = Rotation.from_quat(q[3:7]).as_matrix()
        transform = np.zeros((35, 35))
        transform[:3, :3] = np.eye(3)
        transform[3:6, 3:6] = rotation.T
        transform[self.vindex, np.arange(6, 35)] = 1
        d.qpos[:3] = q[:3]
        d.qpos[3:7] = np.roll(q[3:7], 1)
        d.qpos[self.qindex] = q[7:]
        d.qvel[:] = 0 if v is None else transform @ v
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        if v is not None:
            mujoco.mj_comVel(m, d)
            mujoco.mj_subtreeVel(m, d)
        return transform

    def evaluate(self, q):
        self._state(q)
        return self.data.subtree_com[1].copy()

    def momentum(self, q, v, return_state=False):
        self._state(q, v)
        d = self.data
        if return_state:
            return d.subtree_com[1].copy(), d.subtree_linvel[1].copy(), d.subtree_angmom[1].copy()
        return d.subtree_angmom[1].copy()

    def matrices(self, q):
        transform = self._state(q)
        linear = np.zeros((3, 35))
        angular = np.zeros((3, 35))
        mujoco.mj_jacSubtreeCom(self.model, self.data, linear, 1)
        mujoco.mj_angmomMat(self.model, self.data, angular, 1)
        return linear @ transform, angular @ transform
