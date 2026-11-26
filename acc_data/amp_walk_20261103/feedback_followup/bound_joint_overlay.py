"""Shrink a finite joint correction uniformly to retain joint-limit margin."""
from pathlib import Path
import json,numpy as np,torch
from fabrics_sim.reference.interpolation import MotionInterpolator
from blended_spline import BlendedSplineReference
from joint_correction_overlay import JointCorrectionOverlay
from numpy_leg_kinematics import LegKinematics
p=Path(__file__).parent;name='tenth_tail5_y004';raw=np.load(p/(name+'.npz'));data={k:raw[k] for k in raw.files};source=np.load(p/'eight_smooth6_tube2.control_reference.npz')['source_q'];trajectory=np.load(p/'balance_y004.npz')['control_q'];q=torch.tensor(trajectory,dtype=torch.float64);base=BlendedSplineReference(MotionInterpolator(q),q,stride=6,start_frame=564,ramp_frames=16);overlay=JointCorrectionOverlay(base,torch.tensor(trajectory),torch.tensor(source),p/(name+'.npz'));model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',np.load(p/'eight_smooth6_tube2.failure.npz')['joint_names'])
scale=1.;margin=1e-4
for frame in np.arange(760,870.001,.05):
 b=base.kinematics(frame,0)[0].numpy()[0,7:19];delta=overlay.spline(frame)[:12]
 for j in range(12):
  if delta[j]>1e-12:scale=min(scale,(model.upper[j]-margin-b[j])/delta[j])
  elif delta[j]<-1e-12:scale=min(scale,(model.lower[j]+margin-b[j])/delta[j])
assert 0<scale<=1
# Tiny reserve for curvature between the dense sampling points.
scale=max(0,scale-1e-5);data['correction']=data['correction']*scale
np.savez_compressed(p/(name+'_bounded.npz'),**data);report=dict(uniform_scale=float(scale),checked_frame_spacing=.05,requested_joint_margin_rad=margin,physical_tracking_certified=False);(p/(name+'_bounded.json')).write_text(json.dumps(report,indent=2)+'\n');print(report)
