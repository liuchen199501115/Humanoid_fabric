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
import osqp
import torch
from fabrics_sim.fabrics.dynamics_aware import rotation_error


def skew(v):
    x,y,z=v
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


@torch.no_grad()
def plan_reference(fabric, sampler, modes, fps, names, output, start_frame=100, stride=4):
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
        feet.append(st['transforms'][0,foot_indices,:3].double().numpy())
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
    print(f'Centroidal path solved: {N} intervals, residual {violation:g}; projecting whole-body poses',flush=True)
    import xml.etree.ElementTree as ET
    limits={node.attrib['name']:node.find('limit') for node in ET.parse(fabric.urdf_path).getroot().findall('joint')}
    lower_q=np.array([float(limits[name].attrib['lower']) for name in names[:12]])
    upper_q=np.array([float(limits[name].attrib['upper']) for name in names[:12]])
    pos_columns=[0,1,2]+list(range(7,19));vel_columns=list(range(18))
    poses=[];diagnostics=[]
    for frame in range(count):
        source=sampler(float(frame));q=source.clone()
        if frame>=start_frame:
            original=model.snapshot(source);target_position=original['transforms'][:,foot_indices,:3].clone()
            target_rotation=original['rotations'][:,foot_indices].clone()
            target_com=q.new_tensor(com_path(frame/fps))[None,:2]
            nominal=np.r_[q[0,:3].double().numpy(),np.zeros(3),q[0,7:19].double().numpy()];x=nominal.copy()
            source_rotation=Rotation.from_quat(source[0,3:7].double().numpy())
            lo_x=np.r_[nominal[:3]-[.15,.15,.08],np.full(3,-.10),lower_q]
            hi_x=np.r_[nominal[:3]+[.15,.15,.03],np.full(3,.10),upper_q]
            penalty=np.r_[[.01,.01,1.],np.full(3,.5),np.full(12,.03)]
            for iteration in range(45):
                st=model.snapshot(q)
                pe=(target_position-st['transforms'][:,foot_indices,:3]).flatten()
                re=-rotation_error(st['rotations'][:,foot_indices],target_rotation).flatten()
                ce=(target_com-st['com'][:,:2]).flatten()
                residual=torch.cat((pe,re,ce)).double().numpy()
                jac=torch.cat((st['origin_jac'][:,foot_indices].flatten(1,2),
                               st['angular_jac'][:,foot_indices].flatten(1,2),st['com_jac'][:,:2]),1)[0,:,vel_columns].double().numpy()
                weights=np.r_[np.full(6,1000.),np.full(6,100.),np.full(2,500.)]
                A=jac*weights[:,None];b=residual*weights
                dx=np.linalg.solve(A.T@A+np.diag(penalty**2),A.T@b+penalty**2*(nominal-x))
                scale=max(1.,np.max(np.abs(dx[:3]))/.025,np.max(np.abs(dx[3:]))/.12)
                x=np.clip(x+dx/scale,lo_x,hi_x)
                q[:,pos_columns]=q.new_tensor(np.r_[x[:3],x[6:]])[None]
                q[:,3:7]=q.new_tensor((Rotation.from_rotvec(x[3:6])*source_rotation).as_quat())[None]
                if np.max(np.abs(residual[:6]))<2e-6 and np.max(np.abs(residual[6:]))<2e-5:break
            st=model.snapshot(q)
            diagnostics.append([frame,float((st['transforms'][:,foot_indices,:3]-target_position).abs().max()),
                float(rotation_error(st['rotations'][:,foot_indices],target_rotation).abs().max()),
                float((st['com'][:,:2]-target_com).abs().max())])
            # C2 entry preserves the original initial segment and its derivatives.
            u=min(1.,max(0.,(frame-start_frame)/40.));blend=u**3*(10-15*u+6*u*u)
            q=source+blend*(q-source)
            q[:,3:7]/=q[:,3:7].norm(dim=-1,keepdim=True)
        poses.append(q[0].clone())
        if frame%200==0:print(f'Projected reference {frame}/{count}',flush=True)
    trajectory=torch.stack(poses)
    report=dict(planner='offline linearized centroidal path plus source-foot pose IK',source_frames=count,
        start_frame=start_frame,blend_frames=40,contact_model='optimistic flat sole planning patches; measured WBC contacts unchanged',
        qp_status=result.info.status,qp_residual=violation,peak_com_plan_offset_m=np.max(np.abs(state[:,:3]-refs[:,:3]),axis=0).tolist(),
        peak_ik_position_residual_m=max(r[1] for r in diagnostics),peak_ik_rotation_residual_rad=max(r[2] for r in diagnostics),
        peak_ik_com_xy_residual_m=max(r[3] for r in diagnostics),full_tracking_passed=False)
    Path(output).with_suffix('.centroidal_path.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(Path(output).with_suffix('.centroidal_path.npz'),q=trajectory.numpy(),frame=frames,
        planned_state=state,source_centroidal_state=refs,ik_diagnostics=np.array(diagnostics))
    if report['peak_ik_position_residual_m']>.003 or report['peak_ik_com_xy_residual_m']>.01:
        raise RuntimeError('Centroidal reference IK failed preflight: '+str(report))
    return trajectory
