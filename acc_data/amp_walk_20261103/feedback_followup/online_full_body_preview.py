"""Optional preview feedback, always reprojected through the physical WBC."""
import json
import time
import numpy as np
import torch
from fabrics_sim.fabrics.task_request import TaskRequest
from preview_manifold import difference, integrate


class OnlineFullBodyPreview:
    def __init__(self, planner, path, period=.02, track_wrench=False, reuse_verified=False, follow_plan=False, allow_projection_correction=False):
        self.planner = planner
        self.path = path
        self.period = period
        self.track_wrench = track_wrench
        self.reuse_verified = reuse_verified
        self.follow_plan = follow_plan
        if allow_projection_correction and not follow_plan:raise ValueError("Projection correction requires verified trajectory state feedback")
        self.allow_projection_correction = allow_projection_correction
        self.verified_prediction = None
        self.last_time = -1.
        self.last_mode = None
        self.plan = None
        self.plan_integration = 'continuous'
        self.records = []
        # Trigger a new prediction when evolving contact constraints make the
        # held acceleration unreachable; these are planning thresholds only.
        self.projection_scales = np.r_[[5.] * 3, [20.] * 3, [30.] * 29]

    def observation_failure(self, now, frame, error):
        self.plan=None
        self.last_time=-1.
        self.last_mode=None
        self.records.append(dict(time=now,frame=frame,success=False,status='ObservedPatchUnmatched',reason='contact_observation',solve_seconds=0.,error=str(error)))
        self._write()

    def _write(self):
        self.path.write_text(json.dumps(self.records, indent=2) + '\n')

    def _plan(self, q, v, now, frame, rate, mode, first, reason):
        np.savez_compressed(self.path.with_suffix('.latest_problem.npz'),q=q,v=v,time=now,frame=frame,rate=rate,
                            **{f'problem_{key}':value for key,value in first.items()})
        initial_path=self.path.with_suffix('.initial_problem.npz')
        if not initial_path.exists():
            np.savez_compressed(initial_path,q=q,v=v,time=now,frame=frame,rate=rate,
                                **{f'problem_{key}':value for key,value in first.items()})
        started = time.monotonic()
        result = self.planner.refine(q, v, frame, rate, first)
        self.last_time, self.last_mode = now, mode
        record = dict(time=now, frame=frame, mode=mode, reason=reason,
                      success=result['success'], status=result['status'],
                      linear_violation=result['linear_violation'],
                      tube_violation=result['tube_violation'],
                      solve_seconds=time.monotonic() - started,
                      objective=result['objective'])
        self.records.append(record)
        if result['success']:
            np.savez_compressed(self.path.with_suffix('.latest_successful_plan.npz'),
                                q=q, v=v, time=now, frame=frame, rate=rate, **result)
        if result['success'] and not any(r.get('plan_saved', False) for r in self.records):
            np.savez_compressed(self.path.with_suffix('.first_plan.npz'),
                                q=q, v=v, time=now, frame=frame, rate=rate, **result)
            record['plan_saved'] = True
        if not result['success']:
            np.savez_compressed(self.path.with_suffix('.latest_failure_plan.npz'),
                                q=q,v=v,time=now,frame=frame,rate=rate,
                                **{f'problem_{key}':value for key,value in first.items()})
            previous_path=self.path.with_suffix('.latest_successful_plan.npz')
            if previous_path.exists():
                import shutil
                shutil.copyfile(previous_path,self.path.with_suffix('.latest_failure_previous_plan.npz'))
        if not result['success'] and not self.path.with_suffix('.failure_plan.npz').exists():
            np.savez_compressed(self.path.with_suffix('.failure_plan.npz'),
                                q=q, v=v, time=now, frame=frame, rate=rate,
                                **{f'problem_{key}': value for key, value in first.items()})
            latest=self.path.with_suffix('.latest_successful_plan.npz')
            if latest.exists():
                import shutil
                shutil.copyfile(latest,self.path.with_suffix('.failure_previous_plan.npz'))
        record.update({k: value for k, value in result.items()
                       if k.startswith('nonlinear_') or k in
                       ('refinement_history', 'numerical_velocity_trust_recovered',
                        'warm_started', 'warm_iterations', 'warm_start_fallback','collision_cuts','contact_patch_release_history','initial_patch_departures')})
        self.plan = (q, v, result['first_control'][:35], result['first_control'][35:47]) if result['success'] else None
        if result['success']:
            self.verified_prediction = result
            self.plan_integration = result.get('integration','continuous')
        elif self.reuse_verified:
            from verified_preview_reuse import reusable_control
            reuse=reusable_control(self.verified_prediction,q,v,frame,rate,mode)
            if reuse is not None:
                self.plan=reuse['plan']
                self.plan_integration = self.verified_prediction.get('integration','continuous')
                record.update(reused_verified_prediction=True,reused_stage=reuse['stage'],reused_plan_age_seconds=reuse['age_seconds'])
        self._write()

    def _project(self, solver, q, v, now, kwargs, frame=None, rate=None, mode=None):
        q0, v0, a0, wrench0 = self.plan
        elapsed = now - self.last_time
        if self.follow_plan:
            from verified_preview_reuse import reusable_control
            sample=reusable_control(self.verified_prediction,q.detach().cpu().numpy()[0],v.detach().cpu().numpy()[0],frame,rate,mode)
            if sample is None:
                self.records[-1]['plan_sample_mismatch']=True
                return None,float('inf')
            q0,v0,a0,wrench0=sample['plan'];elapsed=0.
            self.records[-1]['latest_sampled_stage']=sample['stage']
        if self.plan_integration=='semiimplicit':
            from preview_integration import advance
            predicted_q,predicted_v=advance(q0,v0,a0,elapsed)
        else:
            predicted_q = integrate(q0, elapsed * v0 + .5 * elapsed**2 * a0)
            predicted_v = v0 + elapsed * a0
        qn = q.detach().cpu().numpy()[0].astype(float)
        vn = v.detach().cpu().numpy()[0].astype(float)
        target = a0 - 20 * difference(qn, predicted_q) - 8 * (vn - predicted_v)
        desired = q.new_tensor(target)[None]
        identity = torch.eye(35, device=q.device, dtype=q.dtype)[None]
        task = TaskRequest('full_body_preview', identity, torch.zeros_like(desired),
                           desired, identity * 10000.,
                           torch.ones_like(desired, dtype=torch.bool), priority='high')
        modified = dict(kwargs)
        modified.update(task_requests=list(kwargs['task_requests']) + [task],
                        priority_tasks=('full_body_preview',), priority_requests=None,
                        priority_levels=None, priority_rows={}, priority_projections=None)
        if self.track_wrench and wrench0 is not None:
            modified.update(contact_wrench_reference=q.new_tensor(wrench0)[None],
                            contact_wrench_weights=q.new_ones((1, 12)))
        result = solver(**modified)
        error = (result.predicted_acc - desired).abs().detach().cpu().numpy().ravel()
        normalized_error = float(np.max(error / self.projection_scales))
        self.records[-1]['latest_acceleration_projection_max_error'] = float(error.max())
        self.records[-1]['latest_acceleration_projection_normalized_error'] = normalized_error
        return result, normalized_error

    def solve(self, baseline_solver, owner, q, v, now, frame, rate, initial_patch_indices=None, **kwargs):
        baseline = baseline_solver(**kwargs)
        if not bool(baseline.validated.all()):
            return baseline
        if frame >= getattr(self.planner,'final_frame',float('inf')):
            return baseline
        mode = int(kwargs['contact_mode'].item())
        qn = q.detach().cpu().numpy()[0].astype(float)
        vn = v.detach().cpu().numpy()[0].astype(float)
        problem = owner.last_priority_problems[0]
        first = {name: getattr(problem, name).detach().cpu().numpy()[0].astype(float)
                 for name in ('A_eq', 'b_eq', 'G', 'b', 'lb', 'ub')}
        first['mode'] = mode
        if initial_patch_indices is not None:
            first['contact_patch_indices'] = initial_patch_indices
        if now - self.last_time >= self.period - 1e-8 or mode != self.last_mode:
            self._plan(qn, vn, now, frame, rate, mode, first, 'period_or_mode')
        if self.plan is None:
            return baseline
        result, error = self._project(baseline_solver, q, v, now, kwargs, frame, rate, mode)
        if (result is None or not bool(result.validated.all()) or (error > 1. and not self.allow_projection_correction)) and now > self.last_time + 1e-8:
            self.records[-1]['replanned_for_contact_projection'] = True
            self._plan(qn, vn, now, frame, rate, mode, first, 'contact_projection')
            if self.plan is None:
                return baseline
            result, error = self._project(baseline_solver, q, v, now, kwargs, frame, rate, mode)
        if result is not None and bool(result.validated.all()) and (error <= 1. or self.allow_projection_correction):
            if error>1.:self.records[-1]['projection_corrected_by_wbc']=True
            self.records[-1]['applied'] = True
            self._write()
            return result
        self.records[-1]['baseline_fallback_for_projection'] = True
        self.plan = None
        self._write()
        return baseline
