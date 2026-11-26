"""Optional native active-set QP solver with independent physical/KKT checks."""
import numpy as np
import torch

from .admm_qp import QPSolution
from .qp_refinement import _violation


class DAQPBackend:
    def __init__(self, max_iter=5000, *, single_solve=False, single_proximal=0., equilibration_cap=1.):
        import daqp
        self.daqp=daqp
        self.max_iter=int(max_iter)
        self.single_solve=single_solve
        if not np.isfinite(single_proximal) or single_proximal < 0:
            raise ValueError('Single-attempt proximal factor must be finite and nonnegative')
        self.single_proximal=float(single_proximal)
        if not np.isfinite(equilibration_cap) or equilibration_cap < 1:
            raise ValueError('Variable equilibration cap must be finite and at least one')
        self.equilibration_cap=float(equilibration_cap)
        self.last_solve_counts=[]
        self._previous_solutions={}
        self.last_status=[]
        self.last_workspace_reused=[]

    def reset(self, env_ids=None):
        if env_ids is None:self._previous_solutions.clear()
        else:
            indices=set(env_ids.detach().cpu().reshape(-1).tolist())
            for key in list(self._previous_solutions):
                if key[1] in indices:del self._previous_solutions[key]
        self.last_status=[];self.last_workspace_reused=[];self.last_solve_counts=[]

    @torch.no_grad()
    def solve(self, qp, warm_z=None, warm_lam_eq=None, workspace='tracking'):
        if not np.isfinite(self.equilibration_cap) or self.equilibration_cap < 1:
            raise ValueError('Variable equilibration cap must be finite and at least one')
        names=['H','h','G','b','lb','ub']+([] if qp.A_eq is None else ['A_eq','b_eq'])
        batch=len(qp.h);device,dtype=qp.h.device,qp.h.dtype
        # One GPU-to-CPU synchronization for all coefficients, not one per field.
        fields=[getattr(qp,name) for name in names]
        sizes=[value[0].numel() for value in fields]
        packed=torch.cat([value.reshape(batch,-1) for value in fields],dim=1).cpu().double().numpy()
        pieces=np.split(packed,np.cumsum(sizes)[:-1],axis=1)
        arrays={name:part.reshape(tuple(value.shape)) for name,part,value in zip(names,pieces,fields)}
        self.last_status=[];self.last_workspace_reused=[];self.last_solve_counts=[]
        solutions=[];duals=[];objectives=[];accepted=[];iterations=[]
        for index in range(batch):
            p={name:value[index] for name,value in arrays.items()}
            n=len(p['h']);E=p.get('A_eq',np.empty((0,n)));e=p.get('b_eq',np.empty(0))
            H=.5*(p['H']+p['H'].T)
            scale=max(1.,np.abs(H).max(),np.abs(p['h']).max())
            matrix=np.ascontiguousarray(np.vstack((E,p['G'])))
            lower=np.r_[p['lb'],e,np.full(len(p['b']),-np.inf)]
            upper=np.r_[p['ub'],e,p['b']]
            sense=np.zeros(len(upper),dtype=np.int32)
            sense[:n][p['lb']==p['ub']]=5;sense[n:n+len(e)]=5
            full=np.vstack((np.eye(n),matrix))
            scaled_H=np.ascontiguousarray(H/scale);scaled_h=np.ascontiguousarray(p['h']/scale)
            variable_scale = None
            if self.equilibration_cap > 1:
                # x = D*y changes coordinates only. Certify x and its mapped
                # multipliers against the ORIGINAL problem below, never the
                # scaled bounds. Warm solutions are also stored in original units.
                diagonal = np.diag(H)
                peak = max(1e-12, float(diagonal.max(initial=0.)))
                variable_scale = np.minimum(self.equilibration_cap,
                    np.sqrt(peak/np.maximum(diagonal, 1e-12)))
                variable_scale = np.maximum(variable_scale, 1.)
                solve_H = np.ascontiguousarray(scaled_H*variable_scale[:, None]*variable_scale[None, :])
                solve_h = np.ascontiguousarray(scaled_h*variable_scale)
                solve_matrix = np.ascontiguousarray(matrix*variable_scale[None, :])
                solve_lower, solve_upper = lower.copy(), upper.copy()
                solve_lower[:n] /= variable_scale
                solve_upper[:n] /= variable_scale
            key=(workspace,index);previous=self._previous_solutions.get(key)
            x=np.zeros(n);lam=np.zeros(len(upper));flag=-1;ok=False;count=0;reused=False
            polished_dual=None
            solve_count=0
            finite_model=all(np.isfinite(value).all() for name,value in p.items() if name not in ('lb','ub'))
            valid_bounds=(not np.isnan(lower).any() and not np.isnan(upper).any()
                          and np.all(lower<=upper) and (lower<np.inf).all() and (upper>-np.inf).all())
            if (finite_model and valid_bounds
                    and np.linalg.eigvalsh(H)[0]>=-1e-7*max(1.,np.abs(H).max())):
                attempts=[(guess,-1e-8,1e-8) for guess in
                          ([previous,None] if previous is not None else [None])]
                attempts.append((None,1e-8,1e-10))
                # Stronger proximal factorization for nearly dependent locks;
                # its inner objective is iterated to the original KKT system.
                attempts.append((None,1e-4,1e-12))
                attempts.append((None,1e-2,1e-12))
                if self.single_solve:
                    # One native attempt; reject a failed certificate without
                    # cold/proximal retries or a separate active-set repair.
                    attempts=([(previous,self.single_proximal,1e-12)]
                              if self.single_proximal > 0 else attempts[:1])
                numerical_failure=False
                for guess,prox,prox_tolerance in attempts:
                    if prox>0 and not numerical_failure and not (self.single_solve and self.single_proximal>0):
                        continue
                    kwargs={} if guess is None else {'primal_start':guess if variable_scale is None else guess/variable_scale}
                    solve_count+=1
                    if variable_scale is None:
                        x,_,flag,info=self.daqp.solve(scaled_H,scaled_h,matrix,upper,lower,sense,
                            primal_tol=1e-8,dual_tol=1e-10,eps_prox=prox,eta_prox=prox_tolerance,
                            iter_limit=self.max_iter,**kwargs)
                    else:
                        x,_,flag,info=self.daqp.solve(solve_H,solve_h,solve_matrix,solve_upper,solve_lower,sense,
                            primal_tol=1e-8,dual_tol=1e-10,eps_prox=prox,eta_prox=prox_tolerance,
                            iter_limit=self.max_iter,**kwargs)
                        x = x*variable_scale
                        info['lam'] = info['lam'].copy()
                        info['lam'][:n] /= variable_scale
                    count+=int(info['iterations']);lam=info['lam']
                    # Numerical failure can leave an arbitrarily poor iterate
                    # even for a feasible QP. A cold proximal solve must not
                    # depend on that failed iterate being nearly feasible.
                    # Its output still needs the unchanged original KKT checks.
                    # Near-dependent task locks can also cause the automatic
                    # factorization to report -1 on a feasible problem. Retry
                    # the ORIGINAL problem, never accept that failed candidate.
                    numerical_failure |= flag in (-1, -2, -4, -7)
                    # A numerical iteration/time/cycling stop is not itself an
                    # optimality certificate. Accept it only if the SAME full
                    # feasibility, stationarity and complementarity checks pass.
                    # Explicit infeasibility/nonconvexity remains a rejection.
                    if flag in (1,-2,-4,-7) and np.isfinite(x).all() and np.isfinite(lam).all():
                        y=full@x;inequality=sense!=5
                        high=inequality & (lam>0);low=inequality & (lam<0)
                        dual_support=np.isfinite(upper[high]).all() and np.isfinite(lower[low]).all()
                        complementarity=max(np.abs(lam[high]*(upper[high]-y[high])).max(initial=0.),
                                            np.abs(lam[low]*(lower[low]-y[low])).max(initial=0.)) if dual_support else np.inf
                        stationarity=np.abs(scaled_H@x+scaled_h+full.T@lam).max(initial=0.)
                        residual=_violation(p,x)
                        numerical_failure |= residual<=1e-3
                        feasible=residual<=1e-6
                        ok=bool(feasible and stationarity<=1e-8 and complementarity<=1e-7)
                        # Native "solved" can itself be numerically wrong.
                        # Retry the unchanged QP even when that iterate is far
                        # from feasible; proximity only gates active-set repair.
                        numerical_failure |= not ok
                        if not self.single_solve and not ok and residual<=1e-3:
                            # A nearly feasible native candidate is only a
                            # starting guess for repair, never an accepted
                            # command. Re-solve its active set against ORIGINAL
                            # coefficients and require the repair's full KKT
                            # and 1e-6 physical feasibility certificate.
                            from .active_set_predictor import predict_active_set,polish_dual_active_set
                            # A newly activated point contact can change more
                            # than four inequalities at once. Bound the repair
                            # work, but allow that whole active set to settle;
                            # its unchanged KKT certificate decides acceptance.
                            polished=polish_dual_active_set(p,lam)
                            if polished is None:
                                polished=predict_active_set(p,x,max_updates=16)
                            if polished is not None:
                                x,polished_dual=polished
                                ok=True
                    if ok:
                        reused=guess is not None
                        break
            if ok:self._previous_solutions[key]=x.copy()
            else:self._previous_solutions.pop(key,None)
            if not np.isfinite(x).all():x=np.zeros(n)
            solutions.append(x)
            duals.append((polished_dual if polished_dual is not None else lam[n:n+len(e)]*scale)
                         if ok else np.zeros(len(e)))
            objectives.append(.5*x@H@x+p['h']@x);accepted.append(ok);iterations.append(count)
            self.last_status.append(f'DAQP {flag}; '+('KKT polished' if polished_dual is not None else
                                                    'KKT certified' if ok else 'rejected'))
            self.last_workspace_reused.append(reused)
            self.last_solve_counts.append(solve_count)
        tensor=lambda values:torch.as_tensor(np.asarray(values),device=device,dtype=dtype)
        valid=torch.tensor(accepted,device=device,dtype=torch.bool)
        return QPSolution(tensor(solutions),valid,max(iterations),tensor(objectives),
                          tensor(duals) if qp.A_eq is not None else None,valid.clone())
