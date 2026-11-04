"""Audited two-foot source tail with a stopped terminal state."""
import json
import math
import xml.etree.ElementTree as ET
import numpy as np
import torch
from .contact_transfer import smooth_progress
from .source_step import SourceStepCandidate, blend_source_body
from .source_step_rollout import SourceStepRollout
from .interpolation import MotionInterpolator
from .whole_body import WholeBodyReference
from .double_support_audit import audit_stop_reference


class QuinticComJoin:
    def __init__(self,start,end,duration):
        self.start,self.end=np.asarray(start),np.asarray(end)
        self.duration=duration

    def sample(self,time):
        s,v,a=smooth_progress(time,self.duration)
        delta=self.end-self.start
        return self.start+s*delta,v*delta,a*delta


class DoubleSupportSourceReference(SourceStepCandidate):
    @torch.no_grad()
    def __init__(self,fabric,initial,names,source,start_frame,end_frame,duration):
        if (type(start_frame) is not int or type(end_frame) is not int
                or not 0<=start_frame<=end_frame<len(source)
                or not math.isfinite(duration) or duration<=0
                or initial.v.abs().max()>1e-4 or initial.a.abs().max()>1e-3):
            raise ValueError('Source tail needs a stopped boundary, valid interval and positive duration')
        self.fabric,self.model,self.names=fabric,fabric.dynamics_tasks.model,names
        self.initial,self.q=initial,initial.q.clone()
        self.feet=[self.model.kinematics.get_link_index(s+'_ankle_roll_link') for s in ('left','right')]
        legs=[names.index(s+'_'+j+'_joint') for s in ('left','right')
              for j in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
        self.columns,self.positions=[0,1,2]+[6+j for j in legs],[0,1,2]+[7+j for j in legs]
        initial_state=self.model.snapshot(initial.q)
        self.start_feet=initial_state['transforms'][:,self.feet].clone()
        self.end_feet=self.start_feet.clone()
        trajectory=source[start_frame:end_frame+1].clone()
        if len(trajectory)==1:trajectory=trajectory.repeat(2,1)
        trajectory=blend_source_body(trajectory,initial.q)
        self.source=MotionInterpolator(trajectory)
        final_state=self.model.snapshot(source[end_frame:end_frame+1])
        self.preview=QuinticComJoin(initial_state['com'][0].numpy(),final_state['com'][0].numpy(),duration)
        self.duration,self.start_frame,self.end_frame=duration,0,len(trajectory)-1
        self.clearance,self.landing_search_depth,self.support=0.,0.,0

    def progress(self,time):
        if not math.isfinite(time) or not 0<=time<=self.duration+1e-8:
            raise ValueError('Source-tail time outside planned interval')
        return time/self.duration,1/self.duration,0.,smooth_progress(time,self.duration)


class DoubleSupportSourceRollout(SourceStepRollout):
    @torch.no_grad()
    def __init__(self,fabric,initial,names,source,start_frame,end_frame,support,dt,armature,
                 report_path,joint_bounds,acceleration_limits,duration=1.):
        if not math.isfinite(dt) or dt<=0 or not math.isfinite(duration) or duration<=0 or support not in (0,1):
            raise ValueError("Source tail requires a positive control period and valid support")
        count=int(round(duration/dt))
        if count<1 or abs(count*dt-duration)>1e-8:
            raise ValueError('Tail duration must span integer control updates')
        plan=DoubleSupportSourceReference(fabric,initial,names,source,start_frame,end_frame,duration)
        limits={n.attrib['name']:n.find('limit') for n in ET.parse(fabric.urdf_path).getroot().findall('joint')}
        effort=source.new_tensor([float(limits[n].attrib['effort']) for n in names])
        self.rows,report=audit_stop_reference(plan,np.arange(count+1)*dt,armature,effort,acceleration_limits,joint_bounds)
        report.update(scope='double_support_source_tail',start_frame=start_frame,end_frame=end_frame)
        report_path.parent.mkdir(parents=True,exist_ok=True)
        report_path.write_text(json.dumps(report,indent=2)+'\n')
        if not report['sampled_necessary_constraints_passed']:
            raise RuntimeError('Source tail failed physical reference checks; see '+str(report_path))
        for row in self.rows:
            row['frame']=start_frame+(end_frame-start_frame)*smooth_progress(row['time'],duration)[0]
        self.references=[WholeBodyReference(*[source.new_tensor(row[k]) for k in ('q','v','a')]) for row in self.rows]
        self.duration,self.stop_duration,self.swing_duration=duration,duration,0.
        self.dt,self.support,self.end_frame=dt,support,end_frame
        self.is_tail,self.touchdown=True,True
        self.fixed_source_frame=self.recovery_target_feet=self.recovery_source_root=None
        self.goal=plan.start_feet[:,1-support].clone()
        self.source_goal=plan.end_feet[:,1-support].clone()

    def contact_mode(self,forces,sole_min,sole_max):
        forces,sole_min,sole_max=(np.asarray(x,dtype=float) for x in (forces,sole_min,sole_max))
        if any(x.shape!=(2,) or not np.isfinite(x).all() for x in (forces,sole_min,sole_max)):
            raise ValueError('Source tail requires finite contact observations')
        if forces.max()<20. or sole_max.max()>.02:
            raise RuntimeError('Source tail lost its planned two-foot support geometry')
        return 3
