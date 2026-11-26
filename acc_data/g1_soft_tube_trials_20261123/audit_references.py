"""Reference mismatch and prescribed stance-foot motion; no control changes."""
import json, sys
from pathlib import Path
import numpy as np
import torch
import warp as wp
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from blended_spline import BlendedSplineReference
from smooth_clock import smooth_clock
from fabrics_sim.reference.interpolation import MotionInterpolator
from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
with np.load(OUT.parent/'g1_both_paths_20261123/baseline.control_reference.npz') as a:
    control=torch.tensor(a['control_q']);source=torch.tensor(a['source_q']);fps=float(a['fps'])
curves={}
for name,trajectory,stride in [('control',control,6),('tube',source,2),('control_stride2',control,2)]:
    curves[name]=BlendedSplineReference(MotionInterpolator(trajectory,method='pchip'),trajectory,
            stride=stride,start_frame=.65*fps,ramp_frames=.1*fps)
with np.load(OUT.parent/'g1_both_paths_20261123/baseline.failure.npz') as a:
    h={k:a['history_'+k].copy() for k in ('time','frame','controller_contact_plan','whole_body_reference_q','whole_body_reference_v','whole_body_reference_a')}
wp.init()
fabric=G1PoseFabric(batch_size=1,device='cpu',timestep=.002,robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware')
model=fabric.dynamics_tasks.model
plan=json.loads((OUT.parent/'g1_both_paths_20261123/baseline.steps_plan.json').read_text())
points=torch.tensor(plan['collision_points']);radii=torch.tensor(plan['collision_radii'])
links=[model.kinematics.get_link_index(side+'_ankle_roll_link') for side in ('left','right')]
rows=[]
for i in sorted(set(range(0,len(h['time']),10))|{600,670,677}):
    t=float(h['time'][i]);frame=float(h['frame'][i]);_,rate,acc=smooth_clock(t,.9,.35,1.2)
    refs={name:curve.kinematics(frame,fps*rate,fps*acc) for name,curve in curves.items()}
    cq,cv,ca=refs['control'];tq,tv,ta=refs['tube']
    for value,key in zip((cq,cv,ca),('q','v','a')):
        np.testing.assert_allclose(value.numpy(),h['whole_body_reference_'+key][i],atol=2e-4,rtol=2e-5)
    row=dict(time=t,frame=frame,joint_q_rmse=float((cq[:,7:]-tq[:,7:]).square().mean().sqrt()),
        joint_v_rmse=float((cv[:,6:]-tv[:,6:]).square().mean().sqrt()),
        joint_a_rmse=float((ca[:,6:]-ta[:,6:]).square().mean().sqrt()))
    mode=int(h['controller_contact_plan'][i,0]);feet={}
    for name,(q,v,a) in refs.items():
        s=model.snapshot(q)
        linear=(s['origin_jac'][:,links]@v[:,None,:,None]).squeeze(-1)[0]
        omega=(s['angular_jac'][:,links]@v[:,None,:,None]).squeeze(-1)[0]
        speed=linear.norm(dim=-1)
        offset=(s['rotations'][0,links,None]@points[:,:,:,None]).squeeze(-1)
        heights=s['transforms'][0,links,None,2]+offset[:,:,2]-radii
        point_speed=(linear[:,None]+torch.linalg.cross(omega[:,None].expand_as(offset),offset)).norm(dim=-1)
        bottom=heights<=heights.min(dim=1,keepdim=True).values+.002
        bottom_speed=point_speed.masked_fill(~bottom,float('inf')).min(dim=1).values
        angular=(s['angular_jac'][:,links]@v[:,None,:,None]).squeeze(-1).norm(dim=-1)[0]
        feet[name]=dict(lowest_collider_height_m=heights.min(dim=1).values.tolist(),
            slowest_bottom_collider_center_speed_m_s=bottom_speed.tolist(),
            active_bottom_speed_m_s=[float(bottom_speed[j]) for j in range(2) if mode & (1<<j)],ankle_origin_speed_m_s=speed.tolist(),ankle_angular_speed_rad_s=angular.tolist(),
            active_ankle_speed_m_s=[float(speed[j]) for j in range(2) if mode & (1<<j)])
    row.update(planned_contact_mode=mode,feet=feet);rows.append(row)
summary={name:float(max(r[name] for r in rows)) for name in ('joint_q_rmse','joint_v_rmse','joint_a_rmse')}
report=dict(scope='Baseline prefix sampled every 20 ms plus 1.2/1.34/1.354 s. Both samplers use the same measured source clock. Ankle-origin speed is a reference rigid-foot consistency diagnostic, not measured ground slip. Bottom collider-center speed permits foot rotation; sphere-surface rolling is not modelled. Control q/v/a are verified against saved controller history.',peak_control_vs_tube=summary,samples=rows)
(OUT/'reference_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(summary));print(json.dumps([r for r in rows if r['time'] in (1.2,1.34,1.354)],indent=2))
