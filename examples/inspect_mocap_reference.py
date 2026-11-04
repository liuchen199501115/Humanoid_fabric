"""Offline G1 mocap audit and plots. No physics, control, or source mutation."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation


def urdf_fk(q, names, path):
    """Vectorized URDF tree FK, including fixed links and inertial COM offsets."""
    tree = ET.parse(path).getroot()
    links = {x.attrib['name']: x for x in tree.findall('link')}
    joints = list(tree.findall('joint'))
    roots = set(links)-{j.find('child').attrib['link'] for j in joints}
    movable = [j.attrib['name'] for j in joints if j.attrib['type'] != 'fixed']
    if len(roots) != 1 or len(set(names)) != len(names) or set(movable) != set(names):
        raise ValueError('URDF must be one tree with exactly the named actuated joints')
    index = {n:i for i,n in enumerate(names)}
    poses = {roots.pop(): (q[:, :3].copy(), Rotation.from_quat(q[:, 3:7]).as_matrix())}
    while joints:
        pending = []
        for joint in joints:
            parent = joint.find('parent').attrib['link']; child = joint.find('child').attrib['link']
            if parent not in poses:
                pending.append(joint); continue
            origin = joint.find('origin')
            xyz = np.zeros(3) if origin is None else np.fromstring(origin.get('xyz','0 0 0'), sep=' ')
            rpy = np.zeros(3) if origin is None else np.fromstring(origin.get('rpy','0 0 0'), sep=' ')
            pp, pr = poses[parent]
            p = pp+np.einsum('nij,j->ni', pr, xyz)
            r = pr@Rotation.from_euler('xyz',rpy).as_matrix()
            kind = joint.attrib['type']
            if kind != 'fixed':
                axis_node = joint.find('axis')
                axis = np.fromstring('1 0 0' if axis_node is None else axis_node.attrib['xyz'], sep=' ')
                axis = axis/np.linalg.norm(axis)
                value = q[:, 7+index[joint.attrib['name']]]
                if kind in ('revolute', 'continuous'):
                    r = r@Rotation.from_rotvec(value[:, None]*axis).as_matrix()
                elif kind == 'prismatic':
                    p += np.einsum('nij,nj->ni',r,value[:,None]*axis)
                else:
                    raise ValueError('Unsupported URDF joint '+kind)
            poses[child] = (p,r)
        if len(pending) == len(joints): raise ValueError('Disconnected URDF tree')
        joints = pending
    mass = 0.; weighted = np.zeros((len(q),3))
    for name, link in links.items():
        inertial = link.find('inertial')
        if inertial is None: continue
        m = float(inertial.find('mass').attrib['value'])
        origin = inertial.find('origin')
        xyz = np.zeros(3) if origin is None else np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
        p,r = poses[name]; weighted += m*(p+np.einsum('nij,j->ni',r,xyz)); mass += m
    if mass <= 0: raise ValueError('URDF must have positive total mass')
    return poses, weighted/mass, mass


def contact_segments(mask, points, fps):
    """Ankle/sole-proxy displacement within each label run; not physical slip."""
    result=[]
    for side in range(2):
        boundaries=np.flatnonzero(np.diff(np.r_[False,mask[:,side],False].astype(int)))
        for start,stop in zip(boundaries[::2],boundaries[1::2]):
            xy=points[start:stop,side,:2]
            result.append(dict(foot=('left','right')[side],start=int(start),stop_exclusive=int(stop),
                duration_s=float((stop-start)/fps),
                displacement_m=float(np.linalg.norm(xy-xy[0],axis=1).max()),
                path_length_m=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum())))
    return result


def pose_gallery(q, poses, com, urdf, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    edges=[(j.find('parent').attrib['link'],j.find('child').attrib['link'])
           for j in ET.parse(urdf).getroot().findall('joint')]
    frames=sorted(set([0,min(114,len(q)-1),min(129,len(q)-1),len(q)//3,2*len(q)//3,len(q)-1]))
    fig=plt.figure(figsize=(14,10))
    for k,frame in enumerate(frames):
        ax=fig.add_subplot(2,3,k+1,projection='3d')
        for parent,child in edges:
            pts=np.stack((poses[parent][0][frame],poses[child][0][frame]))
            color='tab:blue' if child.startswith('left') else 'tab:orange' if child.startswith('right') else 'gray'
            ax.plot(*pts.T,color=color,linewidth=1.5)
        ax.scatter(*com[frame],color='red',s=20)
        center=q[frame,:3]
        ax.set(xlim=(center[0]-.7,center[0]+.7),ylim=(center[1]-.7,center[1]+.7),
               zlim=(center[2]-.9,center[2]+.6),title=f'Frame {frame}',xlabel='X [m]',ylabel='Y [m]')
        ax.set_box_aspect((1.4,1.4,1.5))
    fig.suptitle('URDF link origins: left blue, right orange, COM red; kinematic poses only')
    fig.subplots_adjust(top=.90,bottom=.08,hspace=.35,wspace=.15);fig.savefig(output,dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('motion',type=Path,help='Trusted local PKL')
    parser.add_argument('--contacts-pkl',type=Path)
    parser.add_argument('--urdf',type=Path,default=Path('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'))
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    from test_g1_two_tasks import G1_JOINT_NAMES,load_motion_trajectory,_MotionUnpickler,as_numpy
    q_tensor,fps=load_motion_trajectory(args.motion);q=q_tensor.numpy().astype(float)
    if len(q)<3: raise ValueError('At least three frames are required')
    with args.motion.open('rb') as stream: raw=_MotionUnpickler(stream).load()
    raw_names=raw.get('dof_names')
    if raw_names is not None and len(set(raw_names)) != len(raw_names): raise ValueError('Duplicate joint names')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    poses,com,mass=urdf_fk(q,G1_JOINT_NAMES,args.urdf)
    feet=np.stack([poses[s+'_ankle_roll_link'][0] for s in ('left','right')],axis=1)
    rotations=np.stack([poses[s+'_ankle_roll_link'][1] for s in ('left','right')],axis=1)
    proxy=feet+np.einsum('nfij,j->nfi',rotations,np.array([0,0,-.035]))
    corners=np.array([[-.05,-.025,-.035],[-.05,.025,-.035],[.12,-.025,-.035],[.12,.025,-.035]])
    corner_z=(feet[:,:,None,:]+np.einsum('nfij,kj->nfki',rotations,corners))[...,2]
    floor=float(np.percentile(corner_z.min(axis=2),1))
    height=corner_z.min(axis=2)-floor
    speed=np.linalg.norm(np.gradient(proxy,1/fps,axis=0,edge_order=2)[...,:2],axis=-1)
    provisional=(height<.03)&(speed<.2)
    contact_raw=raw; provenance='embedded labels (not independently verified)'
    if args.contacts_pkl:
        cq,cfps=load_motion_trajectory(args.contacts_pkl)
        if cfps != fps or cq.shape != q_tensor.shape or not np.allclose(cq.numpy(),q,atol=1e-6,rtol=0):
            raise ValueError('Contact companion does not match motion poses/fps')
        with args.contacts_pkl.open('rb') as stream: contact_raw=_MotionUnpickler(stream).load()
        provenance='matching companion labels (not independently verified)'
    mask=contact_raw.get('contact_sequence')
    if mask is None:
        mask=provisional; provenance='provisional height/speed inference; NOT ground truth'
    else:
        mask=np.asarray(as_numpy(mask))
        if mask.shape != (len(q),2) or not np.isin(mask,[0,1]).all(): raise ValueError('Expected binary contact labels [T,2]')
        if list(contact_raw.get('contact_names',['left_foot','right_foot'])) != ['left_foot','right_foot']:
            raise ValueError('Expected left/right contact ordering')
        mask=mask.astype(bool)
    limits={j.attrib['name']:j.find('limit') for j in ET.parse(args.urdf).getroot().findall('joint')}
    lo=np.array([float(limits[n].get('lower','-inf')) for n in G1_JOINT_NAMES])
    hi=np.array([float(limits[n].get('upper','inf')) for n in G1_JOINT_NAMES])
    violation=np.maximum(np.maximum(lo-q[:,7:],q[:,7:]-hi),0)
    joint_v=np.gradient(q[:,7:],1/fps,axis=0,edge_order=2)
    segments=contact_segments(mask,proxy,fps)
    report=dict(scope='Offline reference geometry; no physics or controller validation',
        motion=str(args.motion),sha256=hashlib.sha256(args.motion.read_bytes()).hexdigest(),
        urdf=str(args.urdf),urdf_sha256=hashlib.sha256(args.urdf.read_bytes()).hexdigest(),
        keys=sorted(raw),frames=len(q),fps=fps,sample_span_s=(len(q)-1)/fps,
        q_dimension=36,velocity_dimension=35,actuated_joints=29,total_urdf_mass_kg=mass,
        joint_order_status='verified by supplied names' if raw_names is not None else 'ASSUMED loader order; source names absent',
        joint_mapping=[dict(target_index=i,name=n,source_index=None if raw_names is None else list(raw_names).index(n)) for i,n in enumerate(G1_JOINT_NAMES)],
        conventions=dict(quaternion='loader assumes xyzw; norm alone cannot prove ordering',
            position='metres assumed from URDF/loader; source file unit metadata must confirm',
            joint_angle='radians assumed from URDF revolute joints',world='Z up assumed; absolute yaw/origin not inferred',
            geometry='URDF FK; sole rectangle and -0.035 m offset are controller approximations'),
        raw_quaternion_norm_range=[float(x) for x in (np.linalg.norm(as_numpy(raw['root_rot']),axis=1).min(),np.linalg.norm(as_numpy(raw['root_rot']),axis=1).max())],
        root_position_min=q[:,:3].min(0).tolist(),root_position_max=q[:,:3].max(0).tolist(),
        joint_limit_violating_frames=int((violation>1e-6).any(1).sum()),max_joint_limit_violation_rad=float(violation.max()),
        contact_provenance=provenance,estimated_floor_z_m=floor,
        contact_mode_counts={str(i):int(((mask[:,0]+2*mask[:,1])==i).sum()) for i in range(4)},
        provisional_contact_disagreement_fraction=float((mask!=provisional).mean()),
        contact_segments=segments,foot_metrics={})
    report['per_joint_max_speed_rad_s']=np.abs(joint_v).max(0).tolist()
    for i,side in enumerate(('left','right')):
        active=mask[:,i];selected=speed[active,i]
        report['foot_metrics'][side]=dict(labelled_frames=int(active.sum()),
            labelled_speed_p95_m_s=float(np.percentile(selected,95)) if len(selected) else None,
            labelled_speed_max_m_s=float(selected.max()) if len(selected) else None,
            labelled_above_3cm_frames=int((active&(height[:,i]>.03)).sum()),
            max_segment_displacement_m=max((r['displacement_m'] for r in segments if r['foot']==side),default=0.))
    from fabrics_sim.reference.interpolation import MotionInterpolator
    sampler=MotionInterpolator(q_tensor)
    derivatives=[sampler.kinematics(i,fps) for i in range(len(q))]
    ref_v=np.concatenate([x[1].numpy() for x in derivatives])
    ref_a=np.concatenate([x[2].numpy() for x in derivatives])
    report['derivatives']=dict(method='Existing analytic PCHIP; C1, acceleration can jump at knots',
        angular_velocity_frame='world',finite=bool(np.isfinite(ref_v).all() and np.isfinite(ref_a).all()),
        peak_joint_speed_rad_s=float(np.abs(ref_v[:,6:]).max()),
        peak_joint_acceleration_rad_s2=float(np.abs(ref_a[:,6:]).max()))
    out=args.output_dir
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    selected_links=['torso_link','left_wrist_yaw_link','right_wrist_yaw_link']
    np.savez_compressed(out/'reference_geometry.npz',time=np.arange(len(q))/fps,q=q,reference_v=ref_v,reference_a=ref_a,com=com,feet=feet,
        sole_proxy=proxy,sole_height=height,foot_speed=speed,contacts=mask,provisional_contacts=provisional,
        **{name+'_position':poses[name][0] for name in selected_links})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=np.arange(len(q))/fps;fig,axes=plt.subplots(3,2,figsize=(15,11),layout='constrained')
    for i,side in enumerate(('left','right')):
        axes[0,0].plot(feet[:,i,0],feet[:,i,1],label=side,linewidth=.6)
        axes[0,1].plot(t,height[:,i],label=side,linewidth=.6)
        axes[1,0].plot(t,speed[:,i],label=side,linewidth=.6)
    axes[0,0].plot(com[:,0],com[:,1],label='COM',linewidth=.6);axes[0,0].axis('equal')
    axes[0,0].set(title='World XY reference [m]',xlabel='X [m]',ylabel='Y [m]');axes[0,0].legend()
    axes[0,1].set(title='Sole corner height above estimated floor [m]');axes[0,1].legend()
    axes[1,0].set(title='Sole proxy horizontal speed [m/s]');axes[1,0].legend()
    axes[1,1].imshow(mask.T,aspect='auto',interpolation='nearest',extent=[0,t[-1],1.5,-.5],vmin=0,vmax=1)
    axes[1,1].set(title='Contact labels (provenance in JSON)',yticks=[0,1],yticklabels=['left','right'])
    axes[2,0].plot(t,com[:,2],label='COM');axes[2,0].plot(t,q[:,2],label='root');axes[2,0].legend();axes[2,0].set(title='World height [m]')
    axes[2,1].plot(t,violation.max(1));axes[2,1].set(title='Maximum joint-limit violation [rad]')
    for ax in axes.flat:
        ax.grid(alpha=.2)
    for ax in [axes[0,1],*axes[1],*axes[2]]:ax.set_xlabel('time [s]')
    fig.suptitle('Mocap reference audit — kinematics only, not a tracking result')
    fig.savefig(out/'overview.png',dpi=150);plt.close(fig)
    pose_gallery(q,poses,com,args.urdf,out/'poses.png')
    print(json.dumps({k:v for k,v in report.items() if k not in ('contact_segments','joint_mapping')},indent=2))


if __name__=='__main__': main()
