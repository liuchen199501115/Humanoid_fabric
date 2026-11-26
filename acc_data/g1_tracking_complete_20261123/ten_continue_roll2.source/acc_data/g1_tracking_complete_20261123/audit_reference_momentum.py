"""Full centroidal force and CoP diagnostic for the runtime reference spline."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_centroidal_geometry import CentroidalGeometry
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
from smooth_clock import smooth_clock

names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist()
urdf=str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf')
model=CentroidalGeometry(urdf,names)
contacts=np.load(OUT/'reference_com_cache.npz')['contacts']
times=np.load(OUT/'reference_com_cache.npz')['time']
def jacobian(q):
    c,J,_=model.evaluate(q);x,y,z=c-q[:3]
    return np.c_[np.eye(3),-np.array([[0,-z,y],[z,0,-x],[-y,x,0]]),J]
def advance(q,v,dt):
    q=q.copy();q[:3]+=dt*v[:3]
    q[3:7]=(Rotation.from_rotvec(dt*v[3:6])*Rotation.from_quat(q[3:7])).as_quat();q[7:]+=dt*v[6:]
    return q
if sys.argv[1]=='validate':
    import warp as wp
    wp.config.kernel_cache_dir='/tmp/g1_warp_cache'
    from fabrics_sim.taskmaps.g1_center_of_mass import G1CenterOfMassTaskMap
    from fabrics_sim.taskmaps.g1_centroidal_momentum import G1CentroidalModel
    task=G1CenterOfMassTaskMap(urdf,1,'cpu');other=G1CentroidalModel(task)
    assert task.robot_kinematics.joint_names==names if hasattr(task.robot_kinematics,'joint_names') else True
    rng=np.random.default_rng(41);q=np.load(OUT/'prepared_combined_root_y50.npz')['control_q'];rows=[]
    for k in [0,640,780,802,850,1000,1999]:
        v=rng.normal(size=35);reference=(other.snapshot(torch.tensor(q[k:k+1]))['momentum_matrix'].detach().numpy()[0]@v)
        actual=model.angular_momentum(q[k],v);error=float(np.linalg.norm(actual-reference))
        rows.append(dict(frame=k,momentum_error=error));assert error<1e-4,(k,actual,reference)
    (OUT/'numpy_momentum_validation.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows);sys.exit()
corners=np.array([[.12,.025,-.035],[.12,-.025,-.035],[-.05,.025,-.035],[-.05,-.025,-.035]])
full='--full' in sys.argv
for name in [a for a in sys.argv[1:] if a!='--full']:
    q=torch.tensor(np.load(OUT/(name+'.npz'))['control_q'])
    sampler=BlendedSplineReference(MotionInterpolator(q),q,stride=6,start_frame=130,ramp_frames=20)
    rows=[];cache=[]
    for k in (range(2000) if full else range(660,1001)):
        _,rate,clockacc=smooth_clock(times[k],.9,.35,1.2,2.3,.25,.8,3.,.2,1.2)
        q,v,a=[x.numpy()[0].astype(float) for x in sampler.kinematics(float(k),200*rate,200*clockacc)]
        eps=1e-4;plus=advance(q,v,eps);minus=advance(q,v,-eps)
        com,_,links=model.evaluate(q)
        acc=jacobian(q)@a+(jacobian(plus)-jacobian(minus))@v/(2*eps)
        Hdot=(model.angular_momentum(plus,v+eps*a)-model.angular_momentum(minus,v-eps*a))/(2*eps)
        F=model.mass*(acc+np.array([0,0,9.81]));cop=None;outside=None;hullpoints=None
        cache.append((com,acc,Hdot,np.array([links[side+'_ankle_roll_link'][0] for side in ('left','right')]),
            np.array([links[side+'_ankle_roll_link'][1] for side in ('left','right')])))
        if F[2]>0 and contacts[k].any():
            cop=com[:2]-com[2]*F[:2]/F[2]+np.array([-Hdot[1],Hdot[0]])/F[2]
            hullpoints=np.concatenate([(links[side+'_ankle_roll_link'][1]@corners.T).T+links[side+'_ankle_roll_link'][0]
                for j,side in enumerate(('left','right')) if contacts[k,j]])[:,:2]
            hull=ConvexHull(hullpoints)
            outside=max(0.,float(np.max(hull.equations[:,:2]@cop+hull.equations[:,2])))
        rows.append(dict(frame=k,com_m=com.tolist(),acceleration_m_s2=acc.tolist(),force_N=F.tolist(),momentum_rate_Nm=Hdot.tolist(),
            cop_m=cop.tolist() if cop is not None else None,cop_outside_m=outside,contacts=contacts[k].tolist()))
    report=dict(candidate=name,assumption='Full reference centroidal angular momentum, ground z=0, optimistic full nominal sole support',
        windows=[],frames=rows,full_tracking_passed=False)
    for lo,hi in [(660,740),(740,780),(780,802),(802,884),(884,1000)]:
        subset=[r for r in rows if lo<=r['frame']<=hi]
        distances=[r['cop_outside_m'] for r in subset if r['cop_outside_m'] is not None]
        report['windows'].append(dict(first_frame=lo,last_frame=hi,negative_force_frames=sum(r['force_N'][2]<0 for r in subset),
            cop_outside_fraction=float(np.mean(np.array(distances)>.01)),cop_outside_median_m=float(np.median(distances)),
            max_momentum_rate_Nm=float(max(np.linalg.norm(r['momentum_rate_Nm']) for r in subset))))
    (OUT/(name+'.momentum_audit.json')).write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k!='frames'},flush=True)
    if full:
        import hashlib
        np.savez_compressed(OUT/(name+'.centroidal_cache.npz'),com=np.array([c[0] for c in cache]),
            acceleration=np.array([c[1] for c in cache]),momentum_rate=np.array([c[2] for c in cache]),
            feet_position=np.array([c[3] for c in cache]),feet_rotation=np.array([c[4] for c in cache]),
            parent_sha256=hashlib.sha256((OUT/(name+'.npz')).read_bytes()).hexdigest())
