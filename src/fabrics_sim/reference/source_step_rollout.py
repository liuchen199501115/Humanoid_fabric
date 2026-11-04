"""Measured-contact source swing with an optional audited landing stop.

Each rollout is bounded; a following swing must be planned explicitly.
This does not certify continuous walking or full source fidelity.
"""
import json
import xml.etree.ElementTree as ET

import numpy as np
import torch

from .footstep import source_grounded_step
from .source_step import SourceStepCandidate
from .source_step_optimization import correct_candidate
from .whole_body import WholeBodyReference


class SourceStepRollout:
    @torch.no_grad()
    def __init__(self, fabric, initial, names, source, start_frame, end_frame, support, dt,
                 armature, report_path, acceleration_limits=None, joint_bounds=None, candidate_options=None, stop_duration=0.,
                 fixed_source_frame=None, maximum_step_distance=None, strict_landing=False):
        self.fixed_source_frame = fixed_source_frame
        if fixed_source_frame is not None and (type(fixed_source_frame) is not int
                or not 0 <= fixed_source_frame < len(source) or start_frame != fixed_source_frame
                or end_frame != fixed_source_frame):
            raise ValueError('Recovery must hold one valid source frame')
        if maximum_step_distance is not None and (not np.isfinite(maximum_step_distance) or maximum_step_distance <= 0):
            raise ValueError('Step distance cap must be finite and positive')
        self.dt, self.support = float(dt), support
        options = dict(duration=1.3, landing_height=.61, body_smoothing_passes=4, swing_phase_blend=.2)
        if candidate_options is not None:
            allowed = set(options) | {'body_join', 'clearance', 'landing_com_fraction', 'terminal_capture', 'pelvis_pitch_offset', 'body_interpolation', 'landing_fraction', 'landing_search_depth', 'landing_search_duration', 'com_preview', 'landing_com_lateral_offset', 'capture_margin'}
            if set(candidate_options)-allowed:
                raise ValueError('Unknown source-step candidate options')
            options.update(candidate_options)
        if strict_landing and (fixed_source_frame is not None or maximum_step_distance is not None
                or options.get('landing_fraction', 1.) != 1. or options.get('landing_search_depth', 0.) != 0.):
            raise ValueError('Strict source landing forbids recovery, stride scaling and vertical search')
        self.duration = float(options['duration'])
        self.swing_duration = self.duration
        self.stop_duration = float(stop_duration)
        if not np.isfinite(self.stop_duration) or self.stop_duration < 0:
            raise ValueError("Stop duration must be finite and nonnegative")
        if self.stop_duration:
            raise ValueError('Full-body source mode excludes the reduced-model landing stop')
        self.end_frame = int(end_frame)
        self.touchdown = False
        self.touchdown_count = 0
        self.ever_clear = False
        model = fabric.dynamics_tasks.model
        index = model.kinematics.get_link_index(('right' if support == 0 else 'left')+'_ankle_roll_link')
        source_state = model.snapshot(source[end_frame:end_frame+1])
        self.recovery_target_feet = self.recovery_source_root = None
        if fixed_source_frame is not None:
            feet_indices = [model.kinematics.get_link_index(side+'_ankle_roll_link') for side in ('left','right')]
            self.recovery_target_feet = source.new_tensor(np.stack([source_grounded_step(p.cpu().numpy())
                for p in source_state['transforms'][0,feet_indices]]))
            self.recovery_source_root = source[fixed_source_frame,:3].clone()
        pose = source_state['transforms'][0,index].cpu().numpy()
        self.goal = source.new_tensor(pose.copy() if strict_landing else source_grounded_step(pose))[None]
        if maximum_step_distance is not None:
            start_pose = model.snapshot(initial.q)['transforms'][0,index]
            distance = float((self.goal[0,:2]-start_pose[:2]).norm())
            options['landing_fraction'] = min(options.get('landing_fraction',1.),
                maximum_step_distance/max(distance,1e-9))
        plan_source = source if fixed_source_frame is None else source[fixed_source_frame:fixed_source_frame+1].repeat(2,1)
        plan = SourceStepCandidate(fabric, initial, names, plan_source,
                                   start_frame if fixed_source_frame is None else 0,
                                   end_frame if fixed_source_frame is None else 1, support, self.goal, **options)
        self.source_goal = self.goal.clone()
        self.goal = plan.landing_pose.clone()
        if strict_landing and not torch.equal(self.goal, self.source_goal):
            raise RuntimeError('Strict source landing target was modified by the planner')
        limits = {n.attrib['name']:n.find('limit') for n in ET.parse(fabric.urdf_path).getroot().findall('joint')}
        velocity = source.new_tensor([float(limits[name].attrib['velocity']) for name in names])
        effort = source.new_tensor([float(limits[name].attrib['effort']) for name in names])
        count = int(round(self.duration/dt))
        if abs(count*dt-self.duration)>1e-8:
            raise ValueError('Source-step duration must be an integer number of control updates')
        if joint_bounds is None:
            joint_bounds = dict(lower=source.new_tensor([float(limits[n].attrib['lower']) for n in names]),
                upper=source.new_tensor([float(limits[n].attrib['upper']) for n in names]),dt=dt,bandwidth=8.)
        self.rows, report = correct_candidate(plan, np.arange(count+1)*dt, velocity, 10,
            progress=lambda row: print('Source-step planning: '+json.dumps(row),flush=True),
            dynamics_limits=dict(armature=armature,effort=effort,joint_bounds=joint_bounds,
                acceleration=5*fabric.accel_limits if acceleration_limits is None else acceleration_limits))
        if fixed_source_frame is not None:
            for row in self.rows:
                row['frame'] = float(fixed_source_frame)
        report.update(strict_landing=bool(strict_landing),fixed_source_frame=fixed_source_frame,maximum_step_distance=maximum_step_distance,
                      max_joint_bound_score=max(r['joint_bound_score'] for r in self.rows),
                      peak_acceleration_ratio=max(r['acceleration_ratio'] for r in self.rows),
                      peak_torque_ratio=max(r['torque_ratio'] for r in self.rows),
                      candidate_options=options, contact_search_depth=plan.landing_search_depth,
                      contact_search_requires_measured_load=True, source_landing_pose=self.source_goal[0].tolist(),
                      planned_landing_pose=self.goal[0].tolist(),
                      landing_deviation_m=float((self.goal[0,:2]-self.source_goal[0,:2]).norm()), scope='fixed_frame_recovery' if fixed_source_frame is not None else 'planned_source_step',simulation_validated=False,
                      full_pkl_tracking=False,start_frame=start_frame,end_frame=end_frame,duration=self.duration)
        from .collision_audit import audit_collision_rows
        report['collision_audit'] = audit_collision_rows(fabric, self.rows, source)
        report['sampled_necessary_constraints_passed'] &= report['collision_audit']['passed']
        worst = max(self.rows, key=lambda row: row['joint_bound_score'])
        from fabrics_sim.solver.joint_control import measured_joint_acceleration_bounds
        lo, hi = measured_joint_acceleration_bounds(source.new_tensor(worst['q'])[:,7:],
            source.new_tensor(worst['v'])[:,6:], joint_bounds['lower'], joint_bounds['upper'],
            dt, barrier_bandwidth=joint_bounds['bandwidth'],
            velocity_damper_time=joint_bounds.get('velocity_damper_time'),
            impact_recovery=joint_bounds.get('impact_recovery',False))
        acc = source.new_tensor(worst['a'])[:,6:]
        violation = torch.maximum(lo-acc,acc-hi).clamp_min(0)
        joint = int(violation.argmax())
        report['worst_joint_bound'] = dict(time=worst['time'],joint=names[joint],
            acceleration=float(acc[0,joint]),lower=float(lo[0,joint]),upper=float(hi[0,joint]))
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report,indent=2)+'\n')
        if not report['sampled_necessary_constraints_passed']:
            raise RuntimeError('Source step rejected: required torque/contact/acceleration/joint-braking/collision bounds failed; see '+str(report_path))
        self.references = [WholeBodyReference(*[source.new_tensor(row[k]) for k in ('q','v','a')]) for row in self.rows]

    def index(self, elapsed):
        if not np.isfinite(elapsed) or elapsed < -1e-8:
            raise ValueError('Invalid source-step elapsed time')
        return min(len(self.rows)-1, max(0,int(round(elapsed/self.dt))))

    def sample(self, elapsed):
        if (getattr(self,"stop_duration",0.) and elapsed > self.swing_duration
                and not self.touchdown):
            raise RuntimeError("Double-support stop requires confirmed measured touchdown")
        if elapsed > self.duration+self.dt+1e-8:
            raise RuntimeError('No landing continuation is planned after this source step')
        return self.references[self.index(elapsed)]

    def frame(self, elapsed):
        return float(self.rows[self.index(elapsed)]['frame'])

    def has_measured_support(self, contacts):
        contacts = np.asarray(contacts,dtype=bool)
        if contacts.shape != (2,):
            raise ValueError('Expected two measured contact flags')
        return bool(contacts[self.support] or (self.touchdown and contacts[1-self.support]))

    def contact_mode(self, forces, sole_min, sole_max):
        forces, sole_min, sole_max = (np.asarray(x,dtype=float) for x in (forces,sole_min,sole_max))
        if any(x.shape!=(2,) or not np.isfinite(x).all() for x in (forces,sole_min,sole_max)):
            raise ValueError('Source step requires finite contact observations for both feet')
        swing = 1-self.support
        self.ever_clear |= bool(sole_min[swing] >= .01)
        loaded = bool(self.ever_clear and forces[swing]>=5. and sole_min[swing]<=.002 and sole_max[swing]<=.002)
        self.touchdown_count = self.touchdown_count+1 if loaded else 0
        if self.touchdown_count >= 3:
            self.touchdown = True
        if self.touchdown and not loaded:
            raise RuntimeError('Source step lost its confirmed flat landing contact')
        # Apply a measured flat contact immediately: waiting for the stable-load
        # counter would command a free swing during the impact. Completion still
        # requires three consecutive loaded samples; a transient is not a landing.
        return 3 if loaded else self.support+1
