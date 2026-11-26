"""Offline necessary force/moment conditions for retiming a fixed pose path.

No physics, predictive controller, or reference export. Whole-sole hulls and
box friction deliberately overestimate available support. A feasible result
is not a full-body or native-contact feasibility certificate.
"""
from pathlib import Path
import json
import pickle
import numpy as np
import torch
from scipy.interpolate import BSpline, CubicSpline
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_g1_com import CenterOfMassModel
from numpy_leg_kinematics import LegKinematics

p = Path(__file__).parent
saved = np.load(p / 'wrench_fixed_ten.npz')
urdf = 'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'
com = CenterOfMassModel(urdf, saved['final_joint_order'])
leg = LegKinematics(urdf, saved['final_joint_order'])
plan = json.loads((p / 'wrench_fixed_ten.steps_plan.json').read_text())
with open('acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl', 'rb') as stream:
    contacts = np.asarray(pickle.load(stream)['contact_sequence'])
frames = np.arange(800., 1000.01, 1.)
s = frames / 200.
knots = np.r_[np.full(4, s[0]), np.arange(s[0]+.04, s[-1]-.001, .04), np.full(4, s[-1])]
count = len(knots)-4
basis = BSpline(knots, np.eye(count), 3)
B, D = basis(s), basis(s, 1)
reports = []
for name in ['turn_recovery_y40', 'wbc_turn_support_half_flat_drop20']:
    trajectory = torch.tensor(np.load(p / (name+'.npz'))['control_q'])
    curve = BlendedSplineReference(MotionInterpolator(trajectory), trajectory,
                                  stride=6, start_frame=130, ramp_frames=20)
    samples = [curve.kinematics(float(frame), 200.) for frame in frames]
    state = np.array([com.momentum(q.numpy()[0], v.numpy()[0], return_state=True)
                      for q, v, _ in samples])
    position, tangent_velocity, tangent_momentum = state[:,0], state[:,1], state[:,2]
    acceleration = CubicSpline(s, tangent_velocity)(s, 1)
    momentum_derivative = CubicSpline(s, tangent_momentum)(s, 1)
    force_matrix = com.mass * (acceleration[:,:,None]*B[:,None,:]
                               + .5*tangent_velocity[:,:,None]*D[:,None,:])
    moment_matrix = momentum_derivative[:,:,None]*B[:,None,:] + .5*tangent_momentum[:,:,None]*D[:,None,:]
    gravity = np.array([0., 0., com.mass*9.81])
    force_rows, force_bounds, cop_rows, cop_bounds, cop_frames = [], [], [], [], []
    for k, frame in enumerate(frames):
        active = contacts[int(frame)].astype(bool)
        if not active.any():
            raise ValueError('Flight interval requires separate zero-force equalities')
        F = force_matrix[k]
        force_rows.extend([-F[2], F[2]])
        force_bounds.extend([gravity[2]-20., 800.*active.sum()-gravity[2]])
        for axis in [0, 1]:
            for sign in [-1., 1.]:
                force_rows.append(sign*F[axis]-.6*F[2])
                force_bounds.append(.6*gravity[2])
        vertices = []
        for side in np.flatnonzero(active):
            pos, R, _ = leg.evaluate(samples[k][0].numpy()[0], int(side))
            vertices.extend((np.array(plan['collision_points'][side])@R.T + pos)[:,:2])
        for nx, ny, offset in ConvexHull(np.array(vertices)).equations:
            c = position[k]
            multiplier = np.array([-nx*c[2], -ny*c[2], nx*c[0]+ny*c[1]+offset])
            row = multiplier@F + np.array([ny,-nx,0.])@moment_matrix[k]
            cop_rows.append(row/gravity[2])
            cop_bounds.append(float(-multiplier@gravity/gravity[2]))
            cop_frames.append(float(frame))
    force_rows, force_bounds = np.array(force_rows), np.array(force_bounds)
    cop_rows, cop_bounds = np.array(cop_rows), np.array(cop_bounds)
    # Preserve the speed and its first two derivatives at the interval ends.
    bounds = [(.36, 3.24)]*count
    for index in [0,1,2,count-3,count-2,count-1]:
        bounds[index] = (1.44,1.44)
    force_only = linprog(np.zeros(count), A_ub=force_rows, b_ub=force_bounds,
                         bounds=bounds, method='highs')
    objective = np.r_[np.zeros(count),1.]
    matrix = np.vstack([np.c_[force_rows,np.zeros(len(force_rows))],
                        np.c_[cop_rows,-np.ones(len(cop_rows))]])
    fit = linprog(objective, A_ub=matrix, b_ub=np.r_[force_bounds,cop_bounds],
                  bounds=bounds+[(0.,None)], method='highs')
    result = dict(reference=name,source_frame_interval=[800,1000],
                  speed_range=[.6,1.8],boundary_speed=1.2,
                  force_only_status=force_only.message,
                  force_only_feasible=bool(force_only.success),
                  force_and_moment_status=fit.message,
                  physical_tracking_certified=False,
                  support_model='Optimistic full projected sole hull; box friction; no joint torque constraints')
    if fit.success:
        u=B@fit.x[:count]
        violation=cop_rows@fit.x[:count]-cop_bounds
        selected=int(np.argmax(violation))
        result.update(minimum_maximum_normalized_moment_slack_m=float(fit.x[-1]),
                      worst_source_frame=cop_frames[selected],
                      optimized_speed_range=np.sqrt([u.min(),u.max()]).tolist(),
                      maximum_linear_constraint_residual=float(max(0.,(matrix@fit.x-np.r_[force_bounds,cop_bounds]).max())))
    reports.append(result)
    print(json.dumps(result),flush=True)
(p/'wbc_path_retiming_audit.json').write_text(json.dumps(reports,indent=2)+'\n')
