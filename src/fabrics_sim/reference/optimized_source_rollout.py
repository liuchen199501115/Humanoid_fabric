"""Bounded online first-step probe rebuilt from the live, synchronized boundary.

The full WBC remains responsible for physical contact and actuation. The inner
sole margin is reported separately from the physical wrench cone. This is not
a four-step certificate or an exact stationary-contact reference at entry.
"""
import json
import numpy as np
import torch
import xml.etree.ElementTree as ET

from .source_step_rollout import SourceStepRollout
from .source_step import SourceStepCandidate
from .source_step_optimization import evaluate_candidate, constraint_score
from .full_body_com_path import EndpointComPath, InteriorComPath, WindowedComPath
from .retiming import LandingRetimedReference
from .collision_audit import audit_collision_rows
from .whole_body import WholeBodyReference


def eligible_for_projection_probe(checks, project_nominal_torque=False):
    """Simulation eligibility only; never change the nominal audit result.

    WBC still enforces its original motor bounds on every control update.
    A torque-infeasible exact reference may be tested as a soft task, with
    tracking error measured in simulation. All other preflight gates remain.
    """
    required = {'velocity', 'torque', 'wrench', 'collision', 'boundary', 'landing', 'swing_ground'}
    if set(checks) != required:
        raise ValueError('Incomplete first-step preflight checks')
    return all(value or (key == 'torque' and project_nominal_torque)
               for key, value in checks.items())


class OptimizedSourceRollout(SourceStepRollout):
    @torch.no_grad()
    def __init__(self, fabric, initial, names, source, dt, armature, effort, parameters_path, report_path, *, project_nominal_torque=False):
        config = json.loads(parameters_path.read_text())
        flags = ('body_parameters_enabled', 'interior_enabled', 'early_com_parameters_enabled', 'late_com_parameters_enabled')
        x = np.asarray(config['best']['parameters'], dtype=float)
        pelvis_parameters = bool(config.get('pelvis_parameters_enabled', False))
        if not all(config.get(k) for k in flags) or x.shape != ((19,) if pelvis_parameters else (17,)) or not np.isfinite(x).all():
            raise ValueError('First-step probe requires the audited 17/19-parameter full-body format')
        options = dict(config['options'])
        if options.get('com_preview') != 'full_body_source' or options.get('landing_fraction') != 1. or options.get('landing_search_depth') != 0.:
            raise ValueError('Only unscaled full-body source landing is supported')
        self.dt, self.support, self.end_frame = float(dt), 0, 129
        self.duration = self.swing_duration = float(options['duration'])
        count = round(self.duration/self.dt)
        if count <= 0 or abs(count*self.dt-self.duration) > 1e-8:
            raise ValueError('Duration must contain an integer number of control steps')
        self.stop_duration = 0.
        self.fixed_source_frame = self.recovery_target_feet = self.recovery_source_root = None
        self.touchdown = self.ever_clear = False
        self.touchdown_count = 0
        model = fabric.dynamics_tasks.model
        foot = model.kinematics.get_link_index('right_ankle_roll_link')
        self.goal = model.snapshot(source[129:130])['transforms'][:, foot].clone()
        self.source_goal = self.goal.clone()
        body = source.clone()
        body[:, 19:] = initial.q[:, 19:]
        options['pelvis_pitch_offset'] = float(x[-1])
        if pelvis_parameters:
            options.update(pelvis_roll_offset=float(x[-4]), pelvis_roll_bump=float(x[-3]))
        plan = SourceStepCandidate(fabric, initial, names, body, 114, 129, 0, self.goal, **options)
        plan.preview = EndpointComPath(plan.preview, [x[0], x[1], x[-2]], [x[2], x[3], 0.], [x[4], x[5], 0.])
        plan.preview = InteriorComPath(plan.preview, [[x[6], x[7], 0.], [x[8], x[9], 0.]])
        plan.preview = WindowedComPath(plan.preview, [x[10], x[11], 0.], 0., .6*self.duration)
        plan.preview = WindowedComPath(plan.preview, x[12:15], .45*self.duration, self.duration)
        plan = LandingRetimedReference(plan, config['landing_rate'])
        limits = {n.attrib['name']: n.find('limit') for n in ET.parse(fabric.urdf_path).getroot().findall('joint')}
        velocity = source.new_tensor([float(limits[n].attrib['velocity']) for n in names])
        self.rows = evaluate_candidate(plan, np.arange(count+1)*dt, velocity, dict(armature=armature, effort=effort))
        self.references = [WholeBodyReference(*(source.new_tensor(row[k]) for k in ('q', 'v', 'a'))) for row in self.rows]
        entry = {k: float((getattr(self.references[0], k)-getattr(initial, k)).abs().max()) for k in ('q','v','a')}
        endpoint_error = float((model.snapshot(self.references[-1].q)['transforms'][:, foot, :3]-self.goal[:, :3]).norm())
        collision = audit_collision_rows(fabric, self.rows, source)
        checks = dict(swing_ground=all(r['swing_sole_min_height']>=-2e-5 for r in self.rows),
                      velocity=max(r['velocity_ratio'] for r in self.rows)<=1.,
                      torque=max(r['torque_ratio'] for r in self.rows)<=1.+1e-5,
                      wrench=max(r['exact_wrench_score'] for r in self.rows)<=1e-5,
                      collision=bool(collision['passed']),
                      boundary=entry['q']<=5e-6 and entry['v']<=1e-4 and entry['a']<=1e-3,
                      landing=endpoint_error<=5e-6 and torch.equal(plan.landing_pose, self.goal))
        eligible = eligible_for_projection_probe(checks, project_nominal_torque)
        report = dict(projection_probe_requested=bool(project_nominal_torque),
            eligible_for_bounded_probe=eligible, online_effort_limits_unchanged=True, scope='live-boundary first-step probe preflight', checks=checks,
            physical_sample_checks_passed=all(checks.values()), conservative_score=constraint_score(self.rows),
            conservative_margin_is_online_wbc_constraint=False,
            entry_error=entry, endpoint_position_error_m=endpoint_error, collision=collision,
            minimum_swing_sole_height_m=min(r['swing_sole_min_height'] for r in self.rows),
            peak_torque_ratio=max(r['torque_ratio'] for r in self.rows),
            peak_wrench_score=max(r['exact_wrench_score'] for r in self.rows),
            support_contact_velocity_max=max(r['support_contact_velocity_max'] for r in self.rows),
            support_contact_acceleration_max=max(r['support_contact_acceleration_max'] for r in self.rows),
            stance_entry_transient_s=.2, initial_acceleration_source='same-tick WBC prediction, not measurement',
            effort_joint_names=list(names), effort_limits=effort.detach().cpu().reshape(-1).tolist(),
            source_landing_pose=self.goal.tolist(), samples=len(self.rows),
            acceleration_and_braking_limits_enabled=False, simulation_validated=False, full_pkl_tracking=False)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2)+'\n')
        if not eligible:
            raise RuntimeError('Live first-step candidate failed physical sample checks: '+str(report_path))
