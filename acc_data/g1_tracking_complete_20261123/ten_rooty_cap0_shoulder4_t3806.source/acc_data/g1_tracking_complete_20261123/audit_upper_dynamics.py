"""Recompute upper-body actuator demand independently of unknown foot forces."""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_centroidal_geometry import CentroidalGeometry
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
from g1_armature import unitree_armatures
from smooth_clock import smooth_clock
whole_body='--whole-body' in sys.argv
if whole_body:sys.argv.remove('--whole-body')
from numpy_leg_kinematics import LegKinematics
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist()
urdf=ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'
model=CentroidalGeometry(str(urdf),names);legs=LegKinematics(str(urdf),names)
first_joint=0 if whole_body else 12
armature=np.asarray(unitree_armatures(names)).ravel()[first_joint:]
limits={j.get('name'):[float(j.find('limit').get(key)) for key in ('effort','velocity')] for j in ET.parse(urdf).getroot().findall('joint') if j.find('limit') is not None}
effort,velocity_limit=np.array([limits[name] for name in names[first_joint:]]).T;effort*=.8
times=np.load(OUT/'reference_com_cache.npz')['time']
contacts=np.load(OUT/'reference_com_cache.npz')['contacts']
for name in sys.argv[1:]:
    q=torch.tensor(np.load(OUT/(name+'.npz'))['control_q']);sampler=BlendedSplineReference(MotionInterpolator(q),q,stride=6,start_frame=130,ramp_frames=20);rows=[]
    for k in range(660,1001):
        _,rate,clockacc=smooth_clock(times[k],.9,.35,1.2,2.3,.25,.8,3.,.2,1.2)
        q,v,a=[x.numpy()[0].astype(float) for x in sampler.kinematics(float(k),200*rate,200*clockacc)]
        generalized=model.inverse_dynamics(q,v,a)
        if whole_body:
            F=generalized[:3].copy();M=generalized[3:6].copy();count=max(1,int(contacts[k].sum()))
            for side in range(2):
                if not contacts[k,side]:continue
                foot,_,Jf=legs.evaluate(q,side)
                generalized[6+side*6:12+side*6]-=Jf.T@np.r_[F,M-np.cross(foot-q[:3],F)]/count
        tau=generalized[6+first_joint:]+armature*a[6+first_joint:]
        rows.append(dict(frame=k,torque=tau.tolist(),velocity=v[6+first_joint:].tolist(),
            single_support=bool(contacts[k].sum()==1),
            maximum_effort_ratio=float(np.max(abs(tau)/effort)),maximum_velocity_ratio=float(np.max(abs(v[6+first_joint:])/velocity_limit))))
    windows=[]
    for lo,hi in [(660,740),(740,780),(780,802),(802,884),(884,1000)]:
        subset=[r for r in rows if lo<=r['frame']<=hi]
        windows.append(dict(first_frame=lo,last_frame=hi,maximum_effort_ratio=max(r['maximum_effort_ratio'] for r in subset),
            maximum_velocity_ratio=max(r['maximum_velocity_ratio'] for r in subset),effort_exceeded_frames=sum(r['maximum_effort_ratio']>1 for r in subset)))
    report=dict(candidate=name,scope=('All joints, prescribed source contacts, equal force sharing in double support' if whole_body else 'Upper joints only, no external upper-body contacts')+', 80 percent URDF motor effort budget and configured armature',
        joint_names=names[first_joint:],effort_budget_nm=effort.tolist(),windows=windows,frames=rows,full_tracking_passed=False)
    suffix='.whole_dynamics_audit.json' if whole_body else '.upper_dynamics_audit.json'
    (OUT/(name+suffix)).write_text(json.dumps(report,indent=2)+'\n');print(name,windows,flush=True)
