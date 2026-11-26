"""A checked active-set prediction for adjacent convex controller QPs.

This optional fast path permits a bounded number of active-set corrections.
Infeasibility or failed optimality checks return None to the normal backend.
"""
import numpy as np
from scipy.linalg import lstsq, qr, solve, LinAlgWarning
import warnings


def polish_dual_active_set(problem, multipliers):
    """Refine native dual-selected rows, then certify the ORIGINAL QP.

    Multipliers use DAQP order: variable bounds, equalities, inequalities.
    Large multipliers can amplify tiny primal roundoff into a failed
    complementarity check. Normalize working rows and remove dependencies
    only from the factorization; all original rows remain in acceptance.
    """
    p=problem;n=len(p['h']);H=.5*(p['H']+p['H'].T)
    if (not np.isfinite(H).all() or not np.isfinite(p['h']).all()
            or np.linalg.eigvalsh(H)[0]<-1e-7*max(1.,abs(H).max())):return None
    E=p.get('A_eq',np.empty((0,n)));e=p.get('b_eq',np.empty(0))
    full=np.vstack((np.eye(n),E,p['G']))
    lo=np.r_[p['lb'],e,np.full(len(p['b']),-np.inf)]
    hi=np.r_[p['ub'],e,p['b']]
    if multipliers.shape!=lo.shape or not np.isfinite(multipliers).all():return None
    equal=lo==hi
    positive=(~equal)&(multipliers>1e-10)
    negative=(~equal)&(multipliers<-1e-10)
    if not np.isfinite(hi[positive]).all() or not np.isfinite(lo[negative]).all():return None
    indices=np.r_[np.flatnonzero(equal),np.flatnonzero(positive),np.flatnonzero(negative)]
    signs=np.r_[np.ones(equal.sum()+positive.sum()),-np.ones(negative.sum())]
    A=full[indices]*signs[:,None]
    rhs=np.r_[hi[equal],hi[positive],-lo[negative]]
    norms=np.linalg.norm(A,axis=1)
    if not len(norms) or (norms<=1e-12).any():return None
    rows=A/norms[:,None];target=rhs/norms
    _,R,pivot=qr(rows.T,pivoting=True,mode='economic')
    diagonal=abs(np.diag(R));rank=int((diagonal>1e-12*diagonal.max(initial=0.)).sum())
    chosen=pivot[:rank];working=rows[chosen]
    scale=max(1.,abs(H).max(),abs(p['h']).max())
    K=np.block([[H/scale,working.T],[working,np.zeros((rank,rank))]])
    b=np.r_[-p['h']/scale,target[chosen]]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',LinAlgWarning)
            solution=solve(K,b,assume_a='sym')
            for _ in range(3):solution+=solve(K,b-K@solution,assume_a='sym')
    except (np.linalg.LinAlgError,LinAlgWarning,ValueError):return None
    if not np.isfinite(solution).all():return None
    x=solution[:n];dual=np.zeros(len(indices))
    dual[chosen]=solution[n:]/norms[chosen]
    native=np.zeros(len(lo));native[indices]=dual*signs
    y=full@x
    violation=max((lo-y).max(initial=0.),(y-hi).max(initial=0.))
    stationarity=abs(H@x/scale+p['h']/scale+full.T@native).max(initial=0.)
    complementarity=abs(dual*(rhs-A@x)).max(initial=0.)
    if (violation>1e-6 or stationarity>1e-8 or complementarity>1e-7
            or dual[equal.sum():].min(initial=0.) < -1e-10):return None
    return x,native[n:n+len(e)]*scale


def predict_active_set(problem, previous, tolerance=1e-6, max_updates=0):
    p = problem
    H = .5*(p['H']+p['H'].T)
    n = len(p['h'])
    if previous is None or previous.shape != (n,) or not np.isfinite(previous).all():
        return None
    # The caller assembles Gram-matrix Hessians in float32. Reject materially
    # nonconvex inputs, allowing only roundoff at that assembly precision.
    scale = max(1., np.abs(H).max())
    if np.linalg.eigvalsh(H)[0] < -1e-7*scale:
        return None
    E = p.get('A_eq', np.empty((0,n)))
    e = p.get('b_eq', np.empty(0))
    fixed = p['lb'] == p['ub']
    identity = np.eye(n)
    E_all = np.vstack((E, identity[fixed]))
    e_all = np.r_[e, p['lb'][fixed]]
    C = np.vstack((p['G'], identity[~fixed], -identity[~fixed]))
    b = np.r_[p['b'], p['ub'][~fixed], -p['lb'][~fixed]]
    finite = np.isfinite(b)
    C, b = C[finite], b[finite]
    active = (np.abs(C@previous-b) <= 1e-3) & (np.linalg.norm(C,axis=1)>1e-12)
    seen=set()
    for attempt in range(max_updates+1):
        key=active.tobytes()
        if key in seen:return None
        seen.add(key)
        A = np.vstack((E_all, C[active]))
        rhs = np.r_[e_all,b[active]]
        K = np.block([[H,A.T],[A,np.zeros((len(A),len(A)))]])
        try:
            solution = lstsq(K, np.r_[-p['h'],rhs], cond=1e-12, lapack_driver='gelsy')[0]
        except np.linalg.LinAlgError:
            return None
        if not np.isfinite(solution).all():
            return None
        x, multipliers = solution[:n], solution[n:]
        # Check ALL original constraints, including dependent equalities and
        # rows omitted from this working set. Only a full KKT check accepts x.
        violation = max(np.abs(E@x-e).max(initial=0.), (p['G']@x-p['b']).max(initial=0.),
                        (p['lb']-x).max(initial=0.), (x-p['ub']).max(initial=0.))
        stationarity = np.abs(H@x+p['h']+A.T@multipliers).max(initial=0.)
        stationarity_limit = 1e-8*max(1.,np.abs(p['h']).max())
        dual=multipliers[len(E_all):]
        consistent=np.abs(A@x-rhs).max(initial=0.)<=tolerance
        if (consistent and violation<=tolerance and stationarity<=stationarity_limit
                and dual.min(initial=0.)>=-1e-7):
            return x, multipliers[:len(E)]
        if attempt==max_updates or not consistent or stationarity>stationarity_limit:
            return None
        # A stale bound can become inactive, or a previously slack inequality
        # can bind. Correct one row, then recheck the entire unchanged problem.
        if dual.min(initial=0.) < -1e-7:
            active[np.flatnonzero(active)[np.argmin(dual)]]=False
        else:
            slack=C@x-b
            inactive=~active
            if not inactive.any() or slack[inactive].max(initial=0.)<=tolerance:
                return None
            active[np.argmax(np.where(inactive,slack,-np.inf))]=True
    return None
