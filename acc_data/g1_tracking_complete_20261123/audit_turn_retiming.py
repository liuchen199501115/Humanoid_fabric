"""Offline exact reference dynamics under candidate nonpredictive clocks.

Single-support wrench is determined by Newton-Euler balance; double-support
leg effort assumes equal sharing. This is not a closed-loop certificate.
"""
import json, sys, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_centroidal_geometry import CentroidalGeometry
from numpy_leg_kinematics import LegKinematics
from blended_spline import BlendedSplineReference
from smooth_clock import smooth_clock
from g1_late_clock import late_speed_transition
from g1_armature import unitree_armatures
from fabrics_sim.reference.interpolation import MotionInterpolator

name=sys.argv[1]
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist()
urdf=ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'
model=CentroidalGeometry(str(urdf),names);legs=LegKinematics(str(urdf),names)
limits={j.get('name'):float(j.find('limit').get('effort')) for j in ET.parse(urdf).getroot().findall('joint') if j.find('limit') is not None}
effort=.8*np.array([limits[n] for n in names]);armature=np.asarray(unitree_armatures(names)).ravel()
contacts=np.load(OUT/'reference_com_cache.npz')['contacts']
q=torch.tensor(np.load(OUT/(name+'.npz'))['control_q'])
sampler=BlendedSplineReference(MotionInterpolator(q),q,stride=6,start_frame=130,ramp_frames=20)
reports=[]
for speed in (1.2,1.,.8):
    def clock(t):
        state=smooth_clock(t,.9,.35,1.2,2.3,.25,.8,3.,.2,1.2)
        return late_speed_transition(state,t,start=3.2,ramp=.4,previous_speed=1.2,speed=speed)
    rows=[]
    for k in range(660,1001,2):
        time=brentq(lambda t:clock(t)[0]*200-k,2.,7.)
        _,rate,acc=clock(time)
        q,v,a=[x.numpy()[0].astype(float) for x in sampler.kinematics(float(k),200*rate,200*acc)]
        tau=model.inverse_dynamics(q,v,a);F=tau[:3].copy();M=tau[3:6].copy()
        for side in range(2):
            if not contacts[k,side]:continue
            foot,_,J=legs.evaluate(q,side)
            tau[6+6*side:12+6*side]-=J.T@np.r_[F,M-np.cross(foot-q[:3],F)]/max(1,int(contacts[k].sum()))
        torque=tau[6:]+armature*a[6:]
        rows.append(dict(frame=k,time=time,force=F.tolist(),effort_ratio=float(np.max(abs(torque)/effort)),torque=torque.tolist()))
    report=dict(speed=speed,start=3.2,ramp=.4,frames=rows,windows=[])
    for lo,hi in ((660,740),(740,780),(780,802),(802,884),(884,1000)):
        sub=[r for r in rows if lo<=r['frame']<=hi]
        report['windows'].append(dict(start=lo,end=hi,min_fz=min(r['force'][2] for r in sub),max_fz=max(r['force'][2] for r in sub),peak_effort_ratio=max(r['effort_ratio'] for r in sub)))
    reports.append(report);print(speed,report['windows'],flush=True)
(OUT/(name+'.retimed_dynamics.json')).write_text(json.dumps(dict(candidate=name,reports=reports,full_tracking_passed=False),indent=2)+'\n')
