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
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist()
urdf=ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'
model=CentroidalGeometry(str(urdf),names);armature=np.asarray(unitree_armatures(names)).ravel()[12:]
limits={j.get('name'):[float(j.find('limit').get(key)) for key in ('effort','velocity')] for j in ET.parse(urdf).getroot().findall('joint') if j.find('limit') is not None}
effort,velocity_limit=np.array([limits[name] for name in names[12:]]).T;effort*=.8
times=np.load(OUT/'reference_com_cache.npz')['time']
for name in sys.argv[1:]:
    q=torch.tensor(np.load(OUT/(name+'.npz'))['control_q']);sampler=BlendedSplineReference(MotionInterpolator(q),q,stride=6,start_frame=130,ramp_frames=20);rows=[]
    for k in range(660,1001):
        _,rate,clockacc=smooth_clock(times[k],.9,.35,1.2,2.3,.25,.8,3.,.2,1.2)
        q,v,a=[x.numpy()[0].astype(float) for x in sampler.kinematics(float(k),200*rate,200*clockacc)]
        tau=model.inverse_dynamics(q,v,a)[18:]+armature*a[18:]
        rows.append(dict(frame=k,torque=tau.tolist(),velocity=v[18:].tolist(),
            maximum_effort_ratio=float(np.max(abs(tau)/effort)),maximum_velocity_ratio=float(np.max(abs(v[18:])/velocity_limit))))
    windows=[]
    for lo,hi in [(660,740),(740,780),(780,802),(802,884),(884,1000)]:
        subset=[r for r in rows if lo<=r['frame']<=hi]
        windows.append(dict(first_frame=lo,last_frame=hi,maximum_effort_ratio=max(r['maximum_effort_ratio'] for r in subset),
            maximum_velocity_ratio=max(r['maximum_velocity_ratio'] for r in subset),effort_exceeded_frames=sum(r['maximum_effort_ratio']>1 for r in subset)))
    report=dict(candidate=name,scope='Upper joints only, no external upper-body contacts, 80 percent URDF motor effort budget and configured armature',
        joint_names=names[12:],effort_budget_nm=effort.tolist(),windows=windows,frames=rows,full_tracking_passed=False)
    (OUT/(name+'.upper_dynamics_audit.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,windows,flush=True)
