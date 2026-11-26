"""Optional preview feedback, always reprojected through the physical WBC."""
import json
import time
import numpy as np
import torch
from fabrics_sim.fabrics.task_request import TaskRequest
from full_body_preview import difference, integrate


class OnlineFullBodyPreview:
    def __init__(self, planner, path, period=.02):
        self.planner = planner
        self.path = path
        self.period = period
        self.last_time = -1.
        self.last_mode = None
        self.plan = None
        self.records = []
        # Trigger a new prediction when evolving contact constraints make the
        # held acceleration unreachable; these are planning thresholds only.
        self.projection_scales = np.r_[[5.] * 3, [20.] * 3, [30.] * 29]

    def _write(self):
        self.path.write_text(json.dumps(self.records, indent=2) + '\n')

    def _plan(self, q, v, now, frame, rate, mode, first, reason):
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
        if result['success'] and not any(r.get('plan_saved', False) for r in self.records):
            np.savez_compressed(self.path.with_suffix('.first_plan.npz'),
                                q=q, v=v, time=now, frame=frame, rate=rate, **result)
            record['plan_saved'] = True
        if not result['success'] and not self.path.with_suffix('.failure_plan.npz').exists():
            np.savez_compressed(self.path.with_suffix('.failure_plan.npz'),
                                q=q, v=v, time=now, frame=frame, rate=rate,
                                **{f'problem_{key}': value for key, value in first.items()})
        record.update({k: value for k, value in result.items()
                       if k.startswith('nonlinear_') or k in
                       ('refinement_history', 'numerical_velocity_trust_recovered')})
        self.plan = (q, v, result['first_control'][:35], result['first_control'][35:47]) if result['success'] else None
        self._write()

    def _project(self, solver, q, v, now, kwargs):
        q0, v0, a0, wrench0 = self.plan
        elapsed = now - self.last_time
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
                        priority_levels=None, priority_rows={}, priority_projections=None,
                        contact_wrench_reference=q.new_tensor(wrench0)[None],
                        contact_wrench_weights=q.new_ones((1, 12)))
        result = solver(**modified)
        error = (result.predicted_acc - desired).abs().detach().cpu().numpy().ravel()
        normalized_error = float(np.max(error / self.projection_scales))
        self.records[-1]['latest_acceleration_projection_max_error'] = float(error.max())
        self.records[-1]['latest_acceleration_projection_normalized_error'] = normalized_error
        return result, normalized_error

    def solve(self, baseline_solver, owner, q, v, now, frame, rate, **kwargs):
        baseline = baseline_solver(**kwargs)
        if not bool(baseline.validated.all()):
            return baseline
        mode = int(kwargs['contact_mode'].item())
        qn = q.detach().cpu().numpy()[0].astype(float)
        vn = v.detach().cpu().numpy()[0].astype(float)
        problem = owner.last_priority_problems[0]
        first = {name: getattr(problem, name).detach().cpu().numpy()[0].astype(float)
                 for name in ('A_eq', 'b_eq', 'G', 'b', 'lb', 'ub')}
        first['mode'] = mode
        if now - self.last_time >= self.period - 1e-8 or mode != self.last_mode:
            self._plan(qn, vn, now, frame, rate, mode, first, 'period_or_mode')
        if self.plan is None:
            return baseline
        result, error = self._project(baseline_solver, q, v, now, kwargs)
        if (not bool(result.validated.all()) or error > 1.) and now > self.last_time + 1e-8:
            self.records[-1]['replanned_for_contact_projection'] = True
            self._plan(qn, vn, now, frame, rate, mode, first, 'contact_projection')
            if self.plan is None:
                return baseline
            result, error = self._project(baseline_solver, q, v, now, kwargs)
        if bool(result.validated.all()) and error <= 1.:
            self.records[-1]['applied'] = True
            self._write()
            return result
        self.records[-1]['baseline_fallback_for_projection'] = True
        self.plan = None
        self._write()
        return baseline
