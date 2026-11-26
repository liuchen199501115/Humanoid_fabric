import sys,unittest
from pathlib import Path
import numpy as np
import torch
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'acc_data/amp_walk_20261103/feedback_followup'))
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
from g1_exported_rotation_maps import exported_rotation_maps


class ExportedRotationMapsTest(unittest.TestCase):
    def test_actual_export_interpolation_derivative(self):
        frames=np.arange(91);q=np.zeros((91,36));q[:,2]=1.
        q[:,3:7]=Rotation.from_euler('xyz',np.stack([.4*np.sin(frames*.06),
            .2*np.cos(frames*.04),frames*.03],axis=-1)).as_quat()
        def sampler(values):
            t=torch.tensor(values,dtype=torch.float64)
            return BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=0,ramp_frames=4)
        base=sampler(q);nodes=np.arange(0,91,6);knots=base.smooth.t
        greville=np.array([np.mean(knots[i+1:i+4]) for i in range(len(nodes))])
        select=np.array([5,6,7]);basis=BSpline(knots,np.eye(len(nodes))[:,select],3)
        frame=35.7;rate=240.;acc=60.
        B=basis(frame);V=basis(frame,1)*rate;A=basis(frame,2)*rate**2+basis(frame,1)*acc
        c=base.smooth(frame)[3:7];v=base.smooth(frame,1)[3:7]*rate
        a=base.smooth(frame,2)[3:7]*rate**2+base.smooth(frame,1)[3:7]*acc
        maps=exported_rotation_maps(c,v,a,base.smooth.c[select,3:7],B,V,A,scale=.1)
        direction=np.array([[.2,-.3,.1],[-.4,.1,.2],[.3,.2,-.1]])
        coefficients=np.zeros((len(nodes),3));coefficients[select]=direction*.1
        delta=np.stack([np.interp(frames,greville,coefficients[:,j]) for j in range(3)],axis=1)
        eps=1e-5;states=[]
        for sign in (1,-1):
            values=q.copy();values[:,3:7]=(Rotation.from_rotvec(sign*eps*delta)*Rotation.from_quat(q[:,3:7])).as_quat()
            states.append([x.numpy()[0] for x in sampler(values).kinematics(frame,rate,acc)])
        rotation=(Rotation.from_quat(states[0][0][3:7])*Rotation.from_quat(states[1][0][3:7]).inv()).as_rotvec()/(2*eps)
        expected=[rotation,*[(states[0][j][3:6]-states[1][j][3:6])/(2*eps) for j in (1,2)]]
        for matrix,actual in zip(maps,expected):
            np.testing.assert_allclose(matrix@direction.ravel(),actual,atol=2e-6,rtol=1e-6)


if __name__=='__main__':unittest.main()
