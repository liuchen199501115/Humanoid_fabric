"""Accept shoulder weights only when a same-state baseline bounds task losses.

This is an instantaneous task-residual filter, not a future stability guarantee.
All rejected solves are discarded before physics; fallback restores solver metadata
by solving the original objective again.
"""
import math
import torch

from g1_shoulder_roll import apply_shoulder_roll_weights


class GuardedShoulderWeights:
    def __init__(self, relative=.02, slack=1.):
        if (not math.isfinite(relative) or relative < 0
                or not math.isfinite(slack) or slack < 0):
            raise ValueError('Guard margins must be finite and nonnegative')
        self.relative, self.slack = relative, slack
        self.diagnostic = self.empty_diagnostic()

    @staticmethod
    def empty_diagnostic():
        return dict(shoulder_guard_attempts=0, shoulder_guard_accepted=False,
                    shoulder_guard_multiplier=1., shoulder_guard_reason='inactive',
                    shoulder_guard_worst_excess=0., shoulder_guard_rejected_candidates=0,
                    shoulder_guard_before_residual=0., shoulder_guard_after_residual=0.)

    @staticmethod
    def constraints(owner):
        problem = owner.last_priority_problems[0] if owner.last_priority_problems else owner.last_problem
        return tuple(getattr(problem,k).detach().clone() if getattr(problem,k) is not None else None
                     for k in ('A_eq','b_eq','G','b','lb','ub'))

    @staticmethod
    def same_constraints(before, after):
        return all((a is None and b is None) or
                   (a is not None and b is not None and torch.equal(a,b)) for a,b in zip(before,after))

    def assess(self, baseline, candidate, joints):
        """Compare each foot and root task separately; protect other joint rows."""
        if not bool(candidate.validated.all()):
            return False, 'candidate_invalid', float('inf'), 0., 0.
        groups = [('com','center_of_mass',None,.05),
                  ('body','body_orientation',None,.25),
                  ('pelvis','source_pelvis_orientation',None,.25),
                  ('left_foot_position','feet_pose',(0,1,2),.05),
                  ('right_foot_position','feet_pose',(6,7,8),.05),
                  ('left_foot_orientation','feet_pose',(3,4,5),.25),
                  ('right_foot_orientation','feet_pose',(9,10,11),.25),
                  ('other_joints','joint_posture',tuple(j for j in range(29) if j not in joints),.5)]
        worst, reason = -float('inf'), 'accepted'
        for label, task, indices, floor in groups:
            if task not in baseline.task_residuals or task not in candidate.task_residuals:
                return False, 'missing_'+task, float('inf'), 0., 0.
            a, b = baseline.task_residuals[task], candidate.task_residuals[task]
            if indices is not None:
                a,b=a[:,list(indices)],b[:,list(indices)]
            if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
                return False, 'nonfinite_residual', float('inf'), 0., 0.
            old_rms, new_rms = a.square().mean(-1).sqrt(), b.square().mean(-1).sqrt()
            allowance = self.relative*old_rms + self.slack*floor
            excess = float((new_rms-old_rms-allowance).max())
            if excess > worst:
                worst,reason=excess,label
        old = baseline.task_residuals['joint_posture'][:,list(joints)].square().mean(-1).sqrt()
        new = candidate.task_residuals['joint_posture'][:,list(joints)].square().mean(-1).sqrt()
        before,after=float(old.max()),float(new.max())
        if worst > 1e-6:
            return False, reason, worst, before, after
        if not bool((new <= .99*old).all()):
            return False, 'insufficient_shoulder_improvement', max(worst,0.), before, after
        return True, 'accepted', max(worst,0.), before, after

    def solve(self, solve, owner, *, joint_names, multiplier, side='both', **kwargs):
        self.diagnostic=self.empty_diagnostic()
        requests=kwargs['task_requests']
        posture=next(r for r in requests if r.name=='joint_posture')
        original_metric=posture.metric
        baseline=solve(**kwargs)
        if multiplier <= 1 or not bool(baseline.validated.all()):
            self.diagnostic['shoulder_guard_reason']='inactive' if multiplier<=1 else 'baseline_invalid'
            return baseline
        constraints=self.constraints(owner)
        names=tuple(joint_names)
        joints=[names.index(s+'_shoulder_roll_joint') for s in ('left','right') if side in (s,'both')]
        accepted=False
        try:
            for scale in (multiplier, math.sqrt(multiplier)):
                posture.metric=original_metric
                apply_shoulder_roll_weights(requests,joint_names,scale,side)
                candidate=solve(**kwargs)
                self.diagnostic['shoulder_guard_attempts']+=1
                ok,reason,excess,before,after=self.assess(baseline,candidate,joints)
                if not self.same_constraints(constraints,self.constraints(owner)):
                    ok,reason=False,'changed_constraints'
                self.diagnostic.update(shoulder_guard_reason=reason,
                    shoulder_guard_worst_excess=excess,
                    shoulder_guard_before_residual=before,shoulder_guard_after_residual=after)
                if ok:
                    accepted=True
                    self.diagnostic.update(shoulder_guard_accepted=True,shoulder_guard_multiplier=scale)
                    return candidate
                self.diagnostic['shoulder_guard_rejected_candidates']+=1
        finally:
            if not accepted:
                posture.metric=original_metric
        # Keep owner.last_problem, recovery flags, task residuals and warm-start
        # state aligned with the output actually sent to the simulator.
        owner.reset()
        fallback=solve(**kwargs)
        self.diagnostic['shoulder_guard_after_residual']=float(
            fallback.task_residuals['joint_posture'][:,joints].square().mean(-1).sqrt().max())
        return fallback
