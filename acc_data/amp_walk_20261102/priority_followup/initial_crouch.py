"""Diagnostic joint-bounded pelvis-height adaptation preserving both foot poses."""
import xml.etree.ElementTree as ET
import torch
from fabrics_sim.fabrics.dynamics_aware import rotation_error

@torch.no_grad()
def lower_pelvis_reference(fabric,source,names,depth=.025,start=-60,ramp=60,end=700):
    result=source.clone();model=fabric.dynamics_tasks.model
    limits={j.attrib['name']:j.find('limit') for j in ET.parse(fabric.urdf_path).getroot().findall('joint')}
    feet=[model.kinematics.get_link_index(s+'_ankle_roll_link') for s in ('left','right')]
    legs=[[names.index(s+'_'+j+'_joint') for j in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')] for s in ('left','right')]
    lows=[source.new_tensor([float(limits[names[j]].attrib['lower']) for j in leg]) for leg in legs]
    highs=[source.new_tensor([float(limits[names[j]].attrib['upper']) for j in leg]) for leg in legs]
    eye=torch.eye(6,device=source.device,dtype=source.dtype)[None]
    worst=0.;worst_angle=0.;relaxed=0
    for i in range(max(0,start+1),len(source)):
        u=min(1.,(i-start)/ramp);fade=min(1.,max(0.,(end-i)/100.));drop=depth*u**3*(10-15*u+6*u*u)*fade**3*(10-15*fade+6*fade*fade)
        if drop<1e-7:continue
        target=model.snapshot(source[i:i+1]);positions=target['transforms'][:,feet,:3].clone();rotations=target['rotations'][:,feet].clone()
        q=source[i:i+1].clone();q[:,2]-=drop
        for side,leg in enumerate(legs):
            cols=[6+j for j in leg];indices=[7+j for j in leg]
            q[:,indices[3]]=q[:,indices[3]].clamp_min(.1)
            for iteration in range(60):
                state=model.snapshot(q)
                error=torch.cat((state['transforms'][:,feet[side],:3]-positions[:,side],rotation_error(state['rotations'][:,feet[side]],rotations[:,side])),-1)
                if float(error.abs().max())<2e-5:
                    worst=max(worst,float(error.abs().max()));break
                jac=torch.cat((state['origin_jac'][:,feet[side]],state['angular_jac'][:,feet[side]]),1)[:,:,cols]
                delta=-(jac.transpose(1,2)@torch.linalg.solve(jac@jac.transpose(1,2)+1e-7*eye,error[...,None])).squeeze(-1)
                delta*=torch.clamp(.1/delta.abs().amax(-1,keepdim=True).clamp_min(1e-8),max=1.)
                q[:,indices]=(q[:,indices]+delta).maximum(lows[side]).minimum(highs[side])
            else:
                # Diagnostic retargeting allowance: keep foot positions, permit <= 0.01 rad orientation change.
                weights=q.new_tensor([1.,1.,1.,.1,.1,.1])
                for _ in range(60):
                    state=model.snapshot(q)
                    error=torch.cat((state['transforms'][:,feet[side],:3]-positions[:,side],rotation_error(state['rotations'][:,feet[side]],rotations[:,side])),-1)
                    jac=torch.cat((state['origin_jac'][:,feet[side]],state['angular_jac'][:,feet[side]]),1)[:,:,cols]
                    J=weights[None,:,None]*jac
                    delta=-(J.transpose(1,2)@torch.linalg.solve(J@J.transpose(1,2)+1e-7*eye,(error*weights)[...,None])).squeeze(-1)
                    delta*=torch.clamp(.05/delta.abs().amax(-1,keepdim=True).clamp_min(1e-8),max=1.)
                    q[:,indices]=(q[:,indices]+delta).maximum(lows[side]).minimum(highs[side])
                state=model.snapshot(q)
                error=torch.cat((state['transforms'][:,feet[side],:3]-positions[:,side],rotation_error(state['rotations'][:,feet[side]],rotations[:,side])),-1)
                if float(error[:,:3].abs().max())>=2e-5 or float(error[:,3:].norm())>=.01:
                    raise RuntimeError(f'Crouch IK failed frame={i}, side={side}, residual={error.tolist()}')
                relaxed+=1;worst=max(worst,float(error[:,:3].abs().max()));worst_angle=max(worst_angle,float(error[:,3:].norm()))
        result[i]=q[0]
    return result,dict(depth_m=depth,start_frame=start,ramp_frames=ramp,end_frame=end,max_pose_ik_residual=worst,
                       peak_joint_rmse_rad=float((result[:,7:]-source[:,7:]).square().mean(-1).sqrt().max()),
                       feet_positions_preserved=True,max_relaxed_orientation_rad=worst_angle,relaxed_orientation_samples=relaxed,source_unchanged=True)
