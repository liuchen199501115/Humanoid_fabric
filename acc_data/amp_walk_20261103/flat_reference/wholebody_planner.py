"""Experimental offline centroidal path and consistent whole-body IK reference.

This plans a control reference, never simulator states. Approximate flat support
patches are only used by the planner; live WBC retains measured contact geometry.
Original source frames, landing targets, and fidelity audits remain independent.
"""
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation
from scipy.optimize import lsq_linear
import osqp
import torch
from fabrics_sim.fabrics.dynamics_aware import rotation_error


def skew(v):
    x,y,z=v
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


@torch.no_grad()
def plan_reference(fabric, sampler, modes, fps, names, output, start_frame=0, stride=4, foot_targets=None):
    modes=np.asarray(modes)
    if modes.ndim==2: modes=modes[:,0].astype(int)+2*modes[:,1].astype(int)
    model=fabric.dynamics_tasks.model
    foot_indices=[model.kinematics.get_link_index(s+'_ankle_roll_link') for s in ('left','right')]
    count=len(modes);frames=np.unique(np.r_[np.arange(start_frame,count,stride),count-1]).astype(int)
    refs=[];feet=[]
    for frame in frames:
        q,v,_=sampler.kinematics(float(frame),fps,0.)
        st=model.snapshot(q)
        refs.append(torch.cat((st['com'],(st['com_jac']@v[...,None]).squeeze(-1),
                               (st['momentum_matrix']@v[...,None]).squeeze(-1)),-1)[0].double().numpy())
        feet.append(st['transforms'][0,foot_indices,:3].double().numpy() if foot_targets is None else foot_targets[0][frame])
    refs=np.asarray(refs);feet=np.asarray(feet);feet[:,:,2]=0.
    N=len(frames)-1;nx=9*(N+1);n=nx+12*N
    mass=float(fabric.center_of_mass_taskmap.link_masses.sum())
    weights=np.array([20000.,20000.,10000.,200.,200.,200.,1000.,1000.,300.])
    diagonal=np.r_[np.tile(weights,N+1),np.tile([1e-5,1e-5,1e-5,1e-3,1e-3,1e-3],2*N)]
    target=np.zeros(n);target[:nx]=refs.ravel()
    row_indices=[];column_indices=[];values=[];lower=[];upper=[];cursor=0
    def add(blocks,lo,hi):
        nonlocal cursor
        rows=next(iter(blocks.values())).shape[0]
        for col,block in blocks.items():
            ii,jj=np.nonzero(block)
            row_indices.extend((ii+cursor).tolist());column_indices.extend((jj+col).tolist());values.extend(block[ii,jj].tolist())
        lower.extend(np.broadcast_to(lo,(rows,)).tolist());upper.extend(np.broadcast_to(hi,(rows,)).tolist());cursor+=rows
    add({0:np.eye(9)},refs[0],refs[0])
    for k in range(N):
        dt=(frames[k+1]-frames[k])/fps;mode=int(modes[frames[k]])
        active=[bool(mode&1),bool(mode&2)];number=sum(active)
        ur=np.zeros(12)
        for side in range(2):
            if active[side]:ur[side*6+2]=mass*9.81/number
        target[nx+12*k:nx+12*(k+1)]=ur
        A=np.eye(9);A[:3,3:6]=np.eye(3)*dt
        B=np.zeros((9,12));offset=np.zeros(9);offset[2]=-.5*dt*dt*9.81;offset[5]=-dt*9.81
        total=ur[:3]+ur[6:9];A[6:9,:3]=dt*skew(total);offset[6:9]=-dt*skew(total)@refs[k,:3]
        for side in range(2):
            j=6*side;B[:3,j:j+3]=np.eye(3)*(.5*dt*dt/mass);B[3:6,j:j+3]=np.eye(3)*(dt/mass)
            B[6:9,j:j+3]=dt*skew(feet[k,side]-refs[k,:3]);B[6:9,j+3:j+6]=np.eye(3)*dt
        add({9*k:-A,9*(k+1):np.eye(9),nx+12*k:-B},offset,offset)
        for side in range(2):
            col=nx+12*k+side*6
            if not active[side]:add({col:np.eye(6)},0.,0.);continue
            add({col:np.array([[0.,0.,1.,0.,0.,0.]])},0.,800.)
            cone=[]
            for sx in [-1.,1.]:
                for sy in [-1.,1.]:cone.append([sx,sy,-.5,0.,0.,0.])
            cone.extend([[0,0,-.025,1,0,0],[0,0,-.025,-1,0,0],
                         [0,0,-.05,0,1,0],[0,0,-.12,0,-1,0],
                         [0,0,-.015,0,0,1],[0,0,-.015,0,0,-1]])
            add({col:np.asarray(cone)},-np.inf,0.)
    # Keep the control COM near the source, independently of later fidelity checks.
    for k in range(N+1):
        add({9*k:np.c_[np.eye(3),np.zeros((3,6))]},refs[k,:3]-[.10,.10,.05],refs[k,:3]+[.10,.10,.05])
    matrix=sparse.csc_matrix((values,(row_indices,column_indices)),shape=(cursor,n))
    lo=np.asarray(lower);hi=np.asarray(upper);shift=matrix@target
    qp=osqp.OSQP();qp.setup(P=sparse.diags(2*diagonal,format='csc'),q=np.zeros(n),A=matrix,
        l=lo-shift,u=hi-shift,eps_abs=1e-6,eps_rel=1e-6,max_iter=100000,polish=True,verbose=False)
    result=qp.solve()
    if result.x is None or result.info.status_val not in (1,2):raise RuntimeError('Centroidal path: '+result.info.status)
    solution=result.x+target;error=matrix@solution
    violation=float(max(np.max(lo-error),np.max(error-hi),0.))
    if violation>.002:raise RuntimeError(f'Centroidal path residual {violation:g}')
    state=solution[:nx].reshape(-1,9);com_path=CubicHermiteSpline(frames/fps,state[:,:3],state[:,3:6])
    from scipy.interpolate import CubicSpline
    momentum_path=CubicSpline(frames/fps,state[:,6:9])
    print(f'Centroidal path solved: {N} intervals, residual {violation:g}; projecting whole-body poses',flush=True)
    import xml.etree.ElementTree as ET
    limits={node.attrib['name']:node.find('limit') for node in ET.parse(fabric.urdf_path).getroot().findall('joint')}
    lower_q=np.array([float(limits[name].attrib['lower']) for name in names])
    upper_q=np.array([float(limits[name].attrib['upper']) for name in names])
    lower_q[[3,9]]=np.maximum(lower_q[[3,9]],.03)
    pos_columns=[0,1,2]+list(range(7,36));vel_columns=list(range(35))
    poses=[];diagnostics=[]
    for frame in range(count):
        source=sampler(float(frame));q=source.clone()
        if frame>=start_frame:
            original=model.snapshot(source);target_position=original['transforms'][:,foot_indices,:3].clone()
            target_rotation=original['rotations'][:,foot_indices].clone()
            if foot_targets is not None:
                target_position=q.new_tensor(foot_targets[0][frame])[None]
                target_rotation=q.new_tensor(foot_targets[1][frame])[None]
            target_com=q.new_tensor(com_path(frame/fps))[None]
            target_momentum=q.new_tensor(momentum_path((frame-.5)/fps))[None]
            def momentum_error(st,pose):
                if not poses:return pose.new_zeros(3)
                prev=poses[-1]
                omega=(Rotation.from_quat(pose[0,3:7].double().numpy())*Rotation.from_quat(prev[3:7].double().numpy()).inv()).as_rotvec()*fps
                velocity=torch.cat(((pose[:,:3]-prev[None,:3])*fps,pose.new_tensor(omega)[None],(pose[:,7:]-prev[None,7:])*fps),-1)
                return (target_momentum-(st['momentum_matrix']@velocity[...,None]).squeeze(-1)).flatten()
            nominal=np.r_[q[0,:3].double().numpy(),np.zeros(3),q[0,7:].double().numpy()];x=nominal.copy()
            source_rotation=Rotation.from_quat(source[0,3:7].double().numpy())
            lo_x=np.r_[nominal[:3]-[.15,.15,.18],np.full(3,-.20),lower_q]
            hi_x=np.r_[nominal[:3]+[.15,.15,.03],np.full(3,.20),upper_q]
            penalty=np.r_[[.01,.01,10.],np.full(3,.5),np.full(29,.03)]
            if poses:
                previous=poses[-1].double().numpy();previous_source=sampler(float(frame-1))[0].double().numpy()
                previous_rotation=(Rotation.from_quat(previous[3:7])*Rotation.from_quat(previous_source[3:7]).inv()).as_rotvec()
                warm=np.r_[previous[:3]+source[0,:3].double().numpy()-previous_source[:3],previous_rotation,previous[7:]+nominal[6:]-previous_source[7:]]
                x=np.clip(warm,lo_x,hi_x)
            else:
                x=np.clip(nominal,lo_x,hi_x);warm=nominal.copy()
            temporal=np.r_[np.full(3,.5),np.full(3,.5),np.full(29,.3)] if poses else np.zeros(35)
            q[:,pos_columns]=q.new_tensor(np.r_[x[:3],x[6:]])[None]
            q[:,3:7]=q.new_tensor((Rotation.from_rotvec(x[3:6])*source_rotation).as_quat())[None]
            for iteration in range(80):
                st=model.snapshot(q)
                pe=(target_position-st['transforms'][:,foot_indices,:3]).flatten()
                re=-rotation_error(st['rotations'][:,foot_indices],target_rotation).flatten()
                ce=(target_com-st['com']).flatten()
                residual=torch.cat((pe,re,ce,momentum_error(st,q))).double().numpy()
                jac=torch.cat((st['origin_jac'][:,foot_indices].flatten(1,2),
                               st['angular_jac'][:,foot_indices].flatten(1,2),st['com_jac'],
                               st['momentum_matrix']*fps if poses else st['momentum_matrix']*0.),1)[0,:,vel_columns].double().numpy()
                weights=np.r_[np.full(6,1000.),np.full(6,100.),[500.,500.,250.],np.full(3,100.)]
                A=jac*weights[:,None];b=residual*weights
                trust=np.r_[np.full(3,.025),np.full(3,.025),np.full(29,.12)]
                dx=lsq_linear(np.vstack((A,np.diag(penalty),np.diag(temporal))),np.r_[b,penalty*(nominal-x),temporal*(warm-x)],
                    bounds=(np.maximum(lo_x-x,-trust),np.minimum(hi_x-x,trust)),method='bvls',tol=1e-9,max_iter=100).x
                scale=1.
                cost=float(np.sum((residual*weights)**2)+np.sum((penalty*(x-nominal))**2)+np.sum((temporal*(x-warm))**2))
                accepted=False
                for line_search in range(12):
                    candidate=np.clip(x+dx/(scale*2**line_search),lo_x,hi_x)
                    trial=q.clone()
                    trial[:,pos_columns]=q.new_tensor(np.r_[candidate[:3],candidate[6:]])[None]
                    trial[:,3:7]=q.new_tensor((Rotation.from_rotvec(candidate[3:6])*source_rotation).as_quat())[None]
                    ts=model.snapshot(trial)
                    rr=torch.cat(((target_position-ts['transforms'][:,foot_indices,:3]).flatten(),
                        -rotation_error(ts['rotations'][:,foot_indices],target_rotation).flatten(),
                        (target_com-ts['com']).flatten(),momentum_error(ts,trial))).double().numpy()
                    trial_cost=float(np.sum((rr*weights)**2)+np.sum((penalty*(candidate-nominal))**2)+np.sum((temporal*(candidate-warm))**2))
                    if trial_cost<=cost+1e-9:
                        x=candidate;q=trial;accepted=True;break
                if not accepted:break
                if np.max(np.abs(residual[:6]))<2e-6 and np.max(np.abs(residual[6:]))<2e-5:break
            st=model.snapshot(q)
            diagnostics.append([frame,float((st['transforms'][:,foot_indices,:3]-target_position).abs().max()),
                float(rotation_error(st['rotations'][:,foot_indices],target_rotation).abs().max()),
                float((st['com']-target_com).abs().max()),float(momentum_error(st,q).abs().max())])
            # C2 entry preserves the original initial segment and its derivatives.
            u=min(1.,max(0.,(frame-start_frame)/40.));blend=1. if foot_targets is not None else u**3*(10-15*u+6*u*u)
            q=source+blend*(q-source)
            q[:,3:7]/=q[:,3:7].norm(dim=-1,keepdim=True)
        poses.append(q[0].clone())
        if frame%200==0:print(f'Projected reference {frame}/{count}',flush=True)
    trajectory=torch.stack(poses)
    report=dict(planner='offline linearized centroidal path plus source-foot pose IK',source_frames=count,
        start_frame=start_frame,blend_frames=0,nominal_com_height_offset_m=0.,contact_model='optimistic flat sole planning patches; measured WBC contacts unchanged',
        qp_status=result.info.status,qp_residual=violation,peak_com_plan_offset_m=np.max(np.abs(state[:,:3]-refs[:,:3]),axis=0).tolist(),
        peak_ik_position_residual_m=max(r[1] for r in diagnostics),peak_ik_rotation_residual_rad=max(r[2] for r in diagnostics),
        peak_ik_com_residual_m=max(r[3] for r in diagnostics),peak_ik_momentum_residual=max(r[4] for r in diagnostics),full_tracking_passed=False)
    Path(output).with_suffix('.centroidal_path.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(Path(output).with_suffix('.centroidal_path.npz'),q=trajectory.numpy(),frame=frames,
        planned_state=state,source_centroidal_state=refs,ik_diagnostics=np.array(diagnostics))
    if report['peak_ik_position_residual_m']>.003 or report['peak_ik_com_residual_m']>.01 or report['peak_ik_momentum_residual']>.02:
        raise RuntimeError('Centroidal reference IK failed preflight: '+str(report))
    return trajectory
