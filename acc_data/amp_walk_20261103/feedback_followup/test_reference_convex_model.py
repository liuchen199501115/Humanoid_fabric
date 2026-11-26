import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from mujoco_centroidal_model import CentroidalModel
from reference_convex_model import ReferenceConvexModel

class ReferenceConvexTests(unittest.TestCase):
    def test_signed_distance_derivative_in_free_and_penetrating_poses(self):
        p=Path(__file__).parent;d=np.load(p/'wrench_fixed_ten.npz');u='src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'
        model=ReferenceConvexModel(CentroidalModel(u,d['final_joint_order']),u)
        guide=np.load(p/'wbc_transfer_trust_fit.npz')['control_q'];rng=np.random.default_rng(18)
        for frame in [827,900]:
            q=guide[frame].astype(float);direction=rng.normal(size=35);direction/=np.linalg.norm(direction);eps=1e-6
            gap,J,indices=model.evaluate(q);values=[]
            for sign in [1,-1]:
                pose=q.copy();pose[:3]+=sign*eps*direction[:3];pose[7:]+=sign*eps*direction[6:]
                pose[3:7]=(Rotation.from_rotvec(sign*eps*direction[3:6])*Rotation.from_quat(q[3:7])).as_quat()
                distances,_,ids=model.evaluate(pose);values.append(dict(zip(ids,distances)))
            derivative=np.array([(values[0][j]-values[1][j])/(2*eps) for j in indices])
            np.testing.assert_allclose(J@direction,derivative,atol=2e-4,rtol=2e-3)
if __name__=='__main__':unittest.main()
