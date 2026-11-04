"""Independent refinement of a finite QP candidate under unchanged constraints."""
import numpy as np
from scipy.linalg import qr, null_space
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize


def refine_qp(problem, initial=None, tolerance=1e-6):
    """Return a converged, feasible SLSQP candidate, or None.

    Linear constraints are exact copies of the original problem. Dependent
    equality rows are removed only for the optimizer, then ALL original rows
    are checked. An LP feasibility point is never returned as an optimized QP.
    """
    p = problem
    A, b = p.get('A_eq', np.empty((0,len(p['h'])))), p.get('b_eq', np.empty(0))
    if initial is None or not np.isfinite(initial).all():
        feasible = linprog(np.zeros(len(p['h'])), A_ub=p['G'], b_ub=p['b'],
                           A_eq=A if len(A) else None, b_eq=b if len(A) else None,
                           bounds=list(zip(p['lb'],p['ub'])), method='highs')
        if not feasible.success:
            return None
        initial = feasible.x
    constraints = [LinearConstraint(p['G'], -np.inf, p['b'])] if len(p['G']) else []
    if len(A):
        rank = np.linalg.matrix_rank(A)
        if rank:
            _, _, indices = qr(A.T, pivoting=True, mode='economic')
            indices = indices[:rank]
            constraints.insert(0, LinearConstraint(A[indices],b[indices],b[indices]))
    scale = max(1., np.abs(p['H']).max(), np.abs(p['h']).max())
    H, h = .5*(p['H']+p['H'].T)/scale, p['h']/scale
    result = minimize(lambda x: .5*x@H@x+h@x, np.clip(initial,p['lb'],p['ub']),
                      jac=lambda x: H@x+h, method='SLSQP',
                      bounds=Bounds(p['lb'],p['ub']), constraints=constraints,
                      options={'maxiter':300, 'ftol':1e-11})
    if not result.success or not np.isfinite(result.x).all():
        return _nullspace_refinement(p, tolerance)
    x = result.x
    if _violation(p, x) <= tolerance:
        return x, result.nit
    return _nullspace_refinement(p, tolerance)


def _violation(p, x):
    A = p.get('A_eq', np.empty((0, len(x))))
    b = p.get('b_eq', np.empty(0))
    violation = max(np.abs(A@x-b).max(initial=0.),
                    (p['G']@x-p['b']).max(initial=0.),
                    (p['lb']-x).max(initial=0.), (x-p['ub']).max(initial=0.))
    return violation


def _nullspace_refinement(p, tolerance):
    """Eliminate dynamics equalities before retrying the same convex QP.

    Fixed inactive-contact variables create inequalities that are constant
    on the equality subspace. Roundoff in those rows can make SLSQP reject a
    feasible problem. Remove only these constant rows from this optimizer;
    acceptance still checks every original constraint in physical units.
    """
    n = len(p['h'])
    A = p.get('A_eq', np.empty((0, n)))
    b = p.get('b_eq', np.empty(0))
    basis = null_space(A, rcond=1e-10) if len(A) else np.eye(n)
    origin = np.linalg.lstsq(A, b, rcond=1e-10)[0] if len(A) else np.zeros(n)
    if np.abs(A@origin-b).max(initial=0.) > tolerance:
        return None
    if not basis.shape[1]:
        return (origin, 0) if _violation(p, origin) <= tolerance else None
    C = np.vstack((p['G'], np.eye(n), -np.eye(n)))
    bounds = np.r_[p['b'], p['ub'], -p['lb']]
    finite = np.isfinite(bounds)
    C, bounds = C[finite], bounds[finite]
    G, rhs = C@basis, bounds-C@origin
    norm = np.linalg.norm(G, axis=1)
    variable = norm > 1e-10
    if (rhs[~variable] < -tolerance).any():
        return None
    G, rhs = G[variable]/norm[variable, None], rhs[variable]/norm[variable]
    feasible = linprog(np.zeros(basis.shape[1]), A_ub=G if len(G) else None,
                       b_ub=rhs if len(G) else None,
                       bounds=[(None, None)]*basis.shape[1], method='highs')
    if not feasible.success:
        return None
    symmetric = .5*(p['H']+p['H'].T)
    H = basis.T@symmetric@basis
    h = basis.T@(symmetric@origin+p['h'])
    scale = max(1., np.abs(H).max(), np.abs(h).max())
    H, h = H/scale, h/scale
    result = minimize(lambda y: .5*y@H@y+h@y, feasible.x,
                      jac=lambda y: H@y+h, method='SLSQP',
                      constraints=[LinearConstraint(G, -np.inf, rhs)] if len(G) else [],
                      options={'maxiter':300, 'ftol':1e-11})
    x = origin+basis@result.x
    if not result.success or not np.isfinite(x).all() or _violation(p, x) > tolerance:
        return None
    return x, result.nit
