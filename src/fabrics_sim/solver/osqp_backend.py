"""CPU OSQP adapter for small physical QPs, preserving Torch result semantics.

The default batched GPU solver is unchanged. No constraints are removed or
relaxed; the caller still validates physical residuals on the returned tensors.
"""
import numpy as np
import torch

from .admm_qp import QPSolution


class OSQPBackend:
    def __init__(self, max_iter=10000, eps_abs=1e-5, eps_rel=1e-5, reuse_workspace=False,
                 active_set_predictor=False):
        import osqp
        self.osqp = osqp
        self.max_iter, self.eps_abs, self.eps_rel = max_iter, eps_abs, eps_rel
        self.last_status = []
        self.reuse_workspace = bool(reuse_workspace)
        self._workspaces = {}
        self.last_workspace_reused = []
        self.active_set_predictor = bool(active_set_predictor)
        self._previous_solutions = {}

    def reset(self, env_ids=None):
        if env_ids is None:
            self._workspaces.clear()
            self._previous_solutions.clear()
        else:
            indices=set(int(i) for i in env_ids.detach().cpu().reshape(-1).tolist())
            for cache in (self._workspaces,self._previous_solutions):
                for key in list(cache):
                    if key[1] in indices:cache.pop(key)
        self.last_status=[]
        self.last_workspace_reused=[]

    @torch.no_grad()
    def solve(self, qp, warm_z=None, warm_lam_eq=None, workspace='tracking'):
        from scipy import sparse
        device, dtype = qp.h.device, qp.h.dtype
        to_numpy = lambda x: x.detach().cpu().double().numpy()
        arrays = {k: to_numpy(getattr(qp, k)) for k in ('H', 'h', 'G', 'b', 'lb', 'ub')}
        has_eq = qp.A_eq is not None
        if has_eq:
            arrays.update(A_eq=to_numpy(qp.A_eq), b_eq=to_numpy(qp.b_eq))
        self.last_status = []
        solutions, objectives, duals, statuses, converged, iterations = [], [], [], [], [], []
        self.last_workspace_reused = []
        for index in range(len(qp.h)):
            p = {k: v[index] for k, v in arrays.items()}
            key = (workspace, index)
            if self.active_set_predictor:
                from .active_set_predictor import predict_active_set
                predicted = predict_active_set(p, self._previous_solutions.get(key), max_updates=4)
                if predicted is not None:
                    x, dual = predicted
                    solutions.append(x)
                    objectives.append(.5*x@p['H']@x+p['h']@x)
                    duals.append(dual)
                    statuses.append(True); converged.append(True); iterations.append(0)
                    self.last_status.append('active-set KKT certified')
                    self.last_workspace_reused.append(False)
                    self._previous_solutions[key] = x.copy()
                    continue
            n = len(p['h'])
            eq = p['A_eq'] if has_eq else np.empty((0, n))
            beq = p['b_eq'] if has_eq else np.empty(0)
            matrix = sparse.vstack((sparse.csc_matrix(eq), sparse.csc_matrix(p['G']),
                                    sparse.eye(n)), format='csc')
            lo = np.r_[beq, np.full(len(p['b']), -np.inf), p['lb']]
            hi = np.r_[beq, p['b'], p['ub']]
            quadratic = sparse.triu(sparse.csc_matrix(p['H']), format='csc')
            def fresh_solver():
                solver = self.osqp.OSQP()
                solver.setup(P=quadratic,
                         q=p['h'], A=matrix, l=lo, u=hi, verbose=False,
                         eps_abs=self.eps_abs, eps_rel=self.eps_rel,
                         # The default 1e-4 infeasibility certificate tolerance
                         # falsely rejects some ill-conditioned, independently
                         # feasible walking QPs. Tighten certificates; actuator
                         # acceptance still uses the caller's physical checks.
                         eps_prim_inf=1e-8, eps_dual_inf=1e-8,
                         max_iter=self.max_iter, polish=True,
                         adaptive_rho=True, adaptive_rho_interval=25)
                return solver
            key = (workspace, index)
            cached = self._workspaces.get(key) if self.reuse_workspace else None
            def same_pattern(a, b):
                return (a.shape == b.shape and np.array_equal(a.indptr, b.indptr)
                        and np.array_equal(a.indices, b.indices))
            def retain_pattern(value, template):
                if value.shape != template.shape or same_pattern(value, template):
                    return value
                # Numerical zeros can appear/disappear in Jacobians at every
                # tick. Retain zero entries in the accumulated sparse layout;
                # all actual coefficients are still copied exactly.
                def keys(a):
                    return np.repeat(np.arange(a.shape[1]), np.diff(a.indptr))*a.shape[0]+a.indices
                old, new = keys(template), keys(value)
                indices = np.searchsorted(old, new)
                subset = (np.all(indices < len(old)) and np.array_equal(old[indices],new))
                if subset:
                    aligned = template.copy()
                    aligned.data.fill(0.)
                    aligned.data[indices] = value.data
                    return aligned
                left, right = template.copy(), value.copy()
                left.data.fill(1.); right.data.fill(1.)
                layout = (left+right).tocsc(); layout.sort_indices()
                indices = np.searchsorted(keys(layout),new)
                layout.data.fill(0.); layout.data[indices] = value.data
                return layout
            fixed = p['lb'] == p['ub']
            active_eq = np.any(eq != 0., axis=1)
            if cached is not None and (not np.array_equal(cached[3],fixed)
                                       or not np.array_equal(cached[4],active_eq)):
                cached = None
            if cached is not None:
                quadratic = retain_pattern(quadratic, cached[1])
                matrix = retain_pattern(matrix, cached[2])
            reused = (cached is not None and same_pattern(cached[1], quadratic)
                      and same_pattern(cached[2], matrix))
            if reused:
                solver = cached[0]
                solver.update(Px=quadratic.data, Ax=matrix.data, q=p['h'], l=lo, u=hi)
            else:
                solver = fresh_solver()
            result = solver.solve()
            extra_iterations = 0
            restarted = False
            # Each priority stage keeps its own primal AND dual state. Never
            # warm the tracking objective with the preceding balance optimum.
            # If cached scaling/warm state fails, reproduce the original cold
            # solve before the existing independent refinement/validation.
            if reused:
                residual = (max((lo-matrix@result.x).max(initial=0.),
                                (matrix@result.x-hi).max(initial=0.))
                            if result.info.status_val in (1, 2) and result.x is not None
                            and np.isfinite(result.x).all() else np.inf)
                if result.info.status_val not in (1, 2) or residual > 1e-3:
                    extra_iterations = result.info.iter
                    solver = fresh_solver()
                    result = solver.solve()
                    restarted = True
            if self.reuse_workspace:
                self._workspaces[key] = (solver, quadratic, matrix, fixed, active_eq)
            self.last_workspace_reused.append(bool(reused and not restarted))
            # Match the existing ADMM contract: at the iteration limit a
            # finite candidate must still pass every caller-side physical
            # residual check. Explicit infeasibility remains a hard rejection.
            eligible = result.info.status_val in (1, 2, self.osqp.constant('OSQP_MAX_ITER_REACHED'))
            ok = eligible and result.x is not None and np.isfinite(result.x).all()
            refined = None
            residual = (max((lo-matrix@result.x).max(initial=0.),
                            (matrix@result.x-hi).max(initial=0.)) if ok else np.inf)
            if residual > 1e-3:
                from .qp_refinement import refine_qp
                refined = refine_qp(p, result.x if ok else None)
            self.last_status.append(result.info.status + ('; cold restart' if restarted else '')
                                    + ('; SLSQP refined' if refined is not None else ''))
            # Infeasibility certificates are not actuator commands. Preserve a
            # finite diagnostic candidate, marked unusable for validation.
            x = refined[0] if refined is not None else result.x if ok else np.zeros(n)
            if self.active_set_predictor:
                if ok or refined is not None:
                    self._previous_solutions[key] = x.copy()
                else:
                    self._previous_solutions.pop(key, None)
            solutions.append(x)
            objectives.append(.5 * x @ p['H'] @ x + p['h'] @ x)
            # SLSQP does not return equality multipliers in the installed
            # SciPy version; this cold-start backend does not reuse them.
            duals.append(result.y[:len(beq)] if ok and refined is None else np.zeros(len(beq)))
            statuses.append(bool(ok or refined is not None))
            converged.append(bool(refined is not None or (ok and result.info.status_val == 1)))
            iterations.append(extra_iterations + result.info.iter + (0 if refined is None else refined[1]))
        tensor = lambda values: torch.as_tensor(np.asarray(values), device=device, dtype=dtype)
        return QPSolution(tensor(solutions), torch.tensor(converged, device=device),
                          max(iterations), tensor(objectives),
                          tensor(duals) if has_eq else None,
                          torch.tensor(statuses, device=device))
