import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'acc_data/g1_tracking_complete_20261123'))
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_centroidal_geometry import CentroidalGeometry
from numpy_leg_kinematics import LegKinematics
from batched_reference_dynamics import reference_outputs


class BatchedDynamicsTests(unittest.TestCase):
    def test_matches_scalar_recursive_dynamics_and_contact_allocation(self):
        urdf=ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'
        all_names=[j.get('name') for j in ET.parse(urdf).getroot().findall('joint') if j.get('type')!='fixed']
        leg_names=[side+'_'+joint+'_joint' for side in ('left','right') for joint in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
        names=leg_names+[n for n in all_names if n not in leg_names]
        model=CentroidalGeometry(urdf,names);legs=LegKinematics(urdf,names)
        rng=np.random.default_rng(94);q=rng.normal(size=(8,36))*.3
        q[:,3:7]=Rotation.random(8,random_state=rng).as_quat()
        v=rng.normal(size=(8,35))*3;a=rng.normal(size=(8,35))*10
        armature=rng.uniform(.001,.05,29)
        contact=np.array([[0,0],[1,0],[0,1],[1,1]]*2,dtype=bool)
        expected=[]
        for qq,vv,aa,cc in zip(q,v,a,contact):
            com=model.evaluate(qq)[0];tau=model.inverse_dynamics(qq,vv,aa)
            F=tau[:3].copy();M=tau[3:6].copy()
            for side in range(2):
                if cc[side]:
                    p,_,J=legs.evaluate(qq,side)
                    tau[6+6*side:12+6*side]-=J.T@np.r_[F,M-np.cross(p-qq[:3],F)]/max(1,int(cc.sum()))
            expected.append(np.r_[F,M-np.cross(com-qq[:3],F),tau[6:]+armature*aa[6:]])
        saved=[x.copy() for x in (q,v,a)]
        np.testing.assert_allclose(reference_outputs(model,q,v,a,contact,armature),expected,rtol=1e-11,atol=1e-10)
        for old,new in zip(saved,(q,v,a)):np.testing.assert_array_equal(old,new)
