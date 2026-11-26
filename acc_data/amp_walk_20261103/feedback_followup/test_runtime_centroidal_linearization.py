"""Independent finite differences through bounded IK and the runtime sampler."""
from pathlib import Path
import numpy as np
import torch
from scipy.interpolate import BSpline,CubicSpline
from scipy.spatial.transform import Rotation
from mujoco_centroidal_model import CentroidalModel
from numpy_g1_com import CenterOfMassModel
from numpy_leg_kinematics import LegKinematics
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
from runtime_centroidal_linearization import linearize,curve_maps

P=Path(__file__).parent
URDF='src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'

def models():
    names=np.load(P/'wrench_fixed_ten.npz')['final_joint_order']
    return CentroidalModel(URDF,names),LegKinematics(URDF,names),CenterOfMassModel(URDF,names)

def sampler(q):
    t=torch.tensor(q,dtype=torch.float64)
    return BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20)

def test_centroidal_independent_urdf_models():
    fast,_,independent=models();q=np.load(P/'wbc_transfer_curve_fit.npz')['control_q'].astype(float)
    rng=np.random.default_rng(14)
    for f in [0,200,799,808,884,952,1000,1050,1500,1999]:
        v=rng.normal(size=35);expected=np.array(independent.momentum(q[f],v,True))
        np.testing.assert_allclose(fast.momentum(q[f],v,True),expected,atol=1e-8,rtol=0)
        J,A=fast.matrices(q[f]);np.testing.assert_allclose(J@v,expected[1],atol=1e-8,rtol=0);np.testing.assert_allclose(A@v,expected[2],atol=1e-8,rtol=0)

def test_composed_maps_match_reconstructed_ik_finite_difference():
    com,leg,_=models();base=np.load(P/'wbc_transfer_curve_fit.npz')['control_q'].astype(float)
    start,end=840,960;knots=np.r_[np.full(4,start),np.arange(start+12,end,12),np.full(4,end)];K=len(knots)-4
    basis=BSpline(knots,np.eye(K),3);coeff=np.zeros((K,23));rng=np.random.default_rng(71)
    direction=rng.normal(size=(K,23))*.01;direction[:3]=0;direction[-3:]=0
    frames=np.unique(np.r_[np.arange(start-4,end+4.01,.5),883.9999,907.9999,951.9999,954.9999]);curve=sampler(base)
    maps=linearize(com,leg,curve,basis,coeff,frames);_,_,D,E=curve_maps(com,leg,curve,basis,coeff,frames)
    runs=[];eps=1e-3
    for sign in [1,-1]:
        out=base.copy()
        for f in range(start+1,end):
            change=sign*eps*(basis(f)@direction);q=base[f].copy();target=com.evaluate(q)+change[:3]
            feet=[leg.evaluate(q,j)[:2] for j in [0,1]]
            q[3:7]=(Rotation.from_rotvec(change[3:6])*Rotation.from_quat(q[3:7])).as_quat();q[19:]+=change[6:]
            for _ in range(40):
                q[:3]+=1.25*(target-com.evaluate(q))
                for side,(pos,R) in enumerate(feet):q,pe,re=solve_leg_pose(leg,q,side,pos,R,position_tolerance=1e-11,orientation_tolerance=1e-11)
                if np.linalg.norm(com.evaluate(q)-target)<1e-11:break
            assert np.linalg.norm(com.evaluate(q)-target)<1e-9
            out[f]=q
        c=sampler(out);samples=[c.kinematics(float(f),240.) for f in frames]
        poses=np.array([s[0].numpy()[0] for s in samples]);vs=np.array([s[1].numpy()[0] for s in samples]);states=np.array([com.momentum(q,v,True) for q,v in zip(poses,vs)])
        runs.append((poses,vs,states))
    dp=(runs[0][0]-runs[1][0])/(2*eps);dv=(runs[0][1]-runs[1][1])/(2*eps);dc=(runs[0][2][:,0]-runs[1][2][:,0])/(2*eps);dh=(runs[0][2][:,2]-runs[1][2][:,2])/(2*eps)
    vector=direction.ravel();times=frames/240.
    np.testing.assert_allclose(D[:,6:]@vector,dp[:,7:],atol=2e-6,rtol=1e-3)
    np.testing.assert_allclose(E@vector,dv,atol=2e-4,rtol=2e-3)
    np.testing.assert_allclose(maps['com']@vector,dc,atol=2e-6,rtol=1e-3)
    np.testing.assert_allclose(maps['momentum']@vector,dh,atol=2e-4,rtol=2e-3)
    np.testing.assert_allclose(maps['acceleration']@vector,CubicSpline(times,(runs[0][2][:,1]-runs[1][2][:,1])/(2*eps))(times,1),atol=.003,rtol=.003)
    np.testing.assert_allclose(maps['momentum_rate']@vector,CubicSpline(times,dh)(times,1),atol=.02,rtol=.003)

import unittest

class RuntimeCentroidalLinearizationTests(unittest.TestCase):
    def test_independent_models(self):
        test_centroidal_independent_urdf_models()

    def test_reconstructed_runtime_derivatives(self):
        test_composed_maps_match_reconstructed_ik_finite_difference()

if __name__ == "__main__":
    unittest.main()
