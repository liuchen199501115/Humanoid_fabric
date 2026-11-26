"""Offline-only conic backend for a bounded convex trajectory QP.

No runtime controller imports this module. The returned multiplier uses the
OSQP convention so stored iterates and independent KKT checks stay comparable.
"""
from types import SimpleNamespace
import numpy as np
from scipy import sparse


def solve_bound_qp(P, q, A, lower, upper, *, tolerance=1e-8, max_iter=200,
                   verbose=False):
    import clarabel

    A = sparse.csc_matrix(A)
    lower, upper = np.asarray(lower), np.asarray(upper)
    if (lower.shape != (A.shape[0],) or upper.shape != lower.shape
            or np.isnan(lower).any() or np.isnan(upper).any()
            or np.any(lower > upper)):
        raise ValueError('Invalid constraint bounds')
    equal = np.isfinite(lower) & (lower == upper)
    hi = np.isfinite(upper) & ~equal
    lo = np.isfinite(lower) & ~equal
    rows = [A[equal], A[hi], -A[lo]]
    rhs = np.r_[upper[equal], upper[hi], -lower[lo]]
    cones = [clarabel.ZeroConeT(int(equal.sum())),
             clarabel.NonnegativeConeT(int(hi.sum() + lo.sum()))]
    settings = clarabel.DefaultSettings()
    settings.max_iter = max_iter
    settings.verbose = verbose
    settings.tol_feas = tolerance
    settings.tol_gap_abs = tolerance
    settings.tol_gap_rel = tolerance
    solver = clarabel.DefaultSolver(sparse.triu(P, format='csc'), np.asarray(q),
        sparse.vstack(rows, format='csc'), rhs, cones, settings)
    result = solver.solve()
    x = np.asarray(result.x)
    z = np.asarray(result.z)
    ne, nh = int(equal.sum()), int(hi.sum())
    y = np.zeros(A.shape[0])
    y[equal] = z[:ne]
    y[hi] += z[ne:ne + nh]
    y[lo] -= z[ne + nh:]
    value = A @ x
    primal = float(np.maximum(np.maximum(lower - value, value - upper), 0).max(initial=0))
    dual = float(np.abs(P @ x + q + A.T @ y).max(initial=0))
    status = str(result.status)
    return SimpleNamespace(x=x, y=y, info=SimpleNamespace(
        status=status, status_val={'Solved': 1, 'AlmostSolved': 2}.get(status, -1),
        obj_val=float(result.obj_val), iter=int(result.iterations),
        prim_res=primal, dual_res=dual,
        native_primal_residual=float(result.r_prim),
        native_dual_residual=float(result.r_dual)))
