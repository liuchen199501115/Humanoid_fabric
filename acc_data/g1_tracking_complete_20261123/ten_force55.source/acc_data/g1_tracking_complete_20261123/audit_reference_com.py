"""Offline COM acceleration and zero-angular-momentum support diagnostic.

The ZMP comparison is a screening approximation, not proof of full-body
infeasibility: angular momentum can change the required center of pressure.
"""
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.spatial import ConvexHull
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from blended_spline import BlendedSplineReference
from smooth_clock import smooth_clock
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_centroidal_geometry import CentroidalGeometry

parent = OUT/'prepared_combined_root_y50.npz'
q = np.load(parent)['control_q']
names = np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist()
model = CentroidalGeometry(str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'), names)
with (ROOT/'acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl').open('rb') as stream:
    contacts = np.asarray(pickle.load(stream)['contact_sequence'], dtype=bool)
sampler = BlendedSplineReference(MotionInterpolator(torch.tensor(q)), torch.tensor(q),
                                 stride=6, start_frame=130, ramp_frames=20)
clock = lambda t: smooth_clock(t, .9, .35, 1.2, 2.3, .25, .8, 3., .2, 1.2)[0]
times = np.array([brentq(lambda t: clock(t)-k/200, 0, 15) for k in range(len(q))])
poses = []; com = []; feet_p = []; feet_R = []
for k in range(len(q)):
    pose = sampler(float(k)).numpy()[0]
    c, _, links = model.evaluate(pose)
    feet = [links[side+'_ankle_roll_link'] for side in ('left', 'right')]
    poses.append(pose); com.append(c); feet_p.append([v[0] for v in feet]); feet_R.append([v[1] for v in feet])
poses, com, feet_p, feet_R = map(np.asarray, (poses, com, feet_p, feet_R))
dtl, dtr = np.diff(times)[:-1], np.diff(times)[1:]
rows = np.repeat(np.arange(len(q)-2), 3)
columns = np.column_stack((np.arange(len(q)-2), np.arange(1,len(q)-1), np.arange(2,len(q)))).ravel()
values = np.column_stack((2/(dtl*(dtl+dtr)), -2/(dtl*dtr), 2/(dtr*(dtl+dtr)))).ravel()
D2 = sparse.csr_matrix((values, (rows, columns)), shape=(len(q)-2,len(q)))
acceleration = D2@com
corners = np.array([[.12,.025,-.035],[.12,-.025,-.035],[-.05,.025,-.035],[-.05,-.025,-.035]])
outside = np.full(len(q)-2, np.nan)
for k in range(1, len(q)-1):
    if not contacts[k].any() or 9.81+acceleration[k-1,2] <= 0:
        continue
    points = np.concatenate([(feet_R[k,s]@corners.T).T+feet_p[k,s] for s in range(2) if contacts[k,s]])[:,:2]
    hull = ConvexHull(points)
    zmp = com[k,:2]-com[k,2]*acceleration[k-1,:2]/(9.81+acceleration[k-1,2])
    outside[k-1] = max(0., float(np.max(hull.equations[:,:2]@zmp+hull.equations[:,2])))
window = np.arange(1,len(q)-1)
report = dict(parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
    mass_kg=model.mass, screening_assumption='Zero centroidal angular momentum rate; full nominal sole footprints. Not a proof of whole-body feasibility.',
    windows=[])
for lo,hi in [(640,740),(740,802),(802,884),(884,1000),(1000,1998)]:
    mask=(window>=lo)&(window<=hi)
    report['windows'].append(dict(first_frame=lo,last_frame=hi,
        minimum_vertical_acceleration=float(acceleration[mask,2].min()),
        maximum_vertical_acceleration=float(acceleration[mask,2].max()),
        negative_required_normal_force_frames=int((acceleration[mask,2]<-9.81).sum()),
        zero_momentum_zmp_outside_fraction=float(np.nanmean(outside[mask]>.01)),
        zero_momentum_zmp_outside_median_m=float(np.nanmedian(outside[mask]))))
np.savez_compressed(OUT/'reference_com_cache.npz', q=poses, com=com, time=times,
    feet_position=feet_p, feet_rotation=feet_R, contacts=contacts, com_acceleration=acceleration,
    zmp_outside=outside, parent_sha256=report['parent_sha256'])
sparse.save_npz(OUT/'reference_com_D2.npz', D2)
(OUT/'reference_com_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
