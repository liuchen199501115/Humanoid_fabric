"""Audit analytic runtime-spline COM acceleration on the critical turn."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_centroidal_geometry import CentroidalGeometry
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist()
model=CentroidalGeometry(str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'),names)
source=np.load(ROOT/'acc_data/g1_tracking_contacts_20261123/eight_verify.control_reference.npz')['source_q']
def sampler(q):
    q=torch.tensor(q)
    return BlendedSplineReference(MotionInterpolator(q),q,stride=6,start_frame=130,ramp_frames=20)
base=sampler(np.load(OUT/'prepared_combined_root_y50.npz')['control_q'])
def jacobian(q):
    c,J,_=model.evaluate(q);x,y,z=c-q[:3]
    return np.c_[np.eye(3),-np.array([[0,-z,y],[z,0,-x],[-y,x,0]]),J]
def advance(q,v,dt):
    q=q.copy();q[:3]+=dt*v[:3]
    q[3:7]=(Rotation.from_rotvec(dt*v[3:6])*Rotation.from_quat(q[3:7])).as_quat()
    q[7:]+=dt*v[6:]
    return q
for name in sys.argv[1:]:
    s=sampler(np.load(OUT/(name+'.npz'))['control_q']);rows=[];max_foot=max_joint=0.
    for k in range(740,885):
        q,v,a=[x.numpy()[0].astype(float) for x in s.kinematics(float(k),240.,0.)]
        eps=1e-4
        acc=jacobian(q)@a+(jacobian(advance(q,v,eps))-jacobian(advance(q,v,-eps)))@v/(2*eps)
        _,_,poses=model.evaluate(q);_,_,old=model.evaluate(base(float(k)).numpy()[0])
        max_foot=max(max_foot,max(float(np.linalg.norm(poses[n][0]-old[n][0])) for n in ('left_ankle_roll_link','right_ankle_roll_link')))
        max_joint=max(max_joint,float(np.sqrt(np.mean((q[7:]-source[k,7:])**2))))
        rows.append(dict(frame=k,vertical_acceleration=float(acc[2]),required_normal_force_N=float(model.mass*(9.81+acc[2]))))
    report=dict(candidate=name,first_frame=740,last_frame=884,
        minimum_vertical_acceleration=min(r['vertical_acceleration'] for r in rows),
        maximum_vertical_acceleration=max(r['vertical_acceleration'] for r in rows),
        negative_required_ground_force_frames=sum(r['required_normal_force_N']<0 for r in rows),
        maximum_foot_position_change_after_runtime_interpolation_m=max_foot,
        peak_source_joint_rmse_rad=max_joint,frames=rows,full_tracking_passed=False)
    (OUT/(name+'.runtime_vertical_audit.json')).write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items() if k!='frames'},flush=True)
