"""Interior-point QP for offline reference fitting, with explicit residual audit."""
from types import SimpleNamespace
import numpy as np
from scipy import sparse
import clarabel

def solve_reference_qp(H,g,C,low,high):
 low=np.asarray(low);high=np.asarray(high);C=sparse.csc_matrix(C)
 equality=np.isfinite(low)&np.isfinite(high)&(abs(low-high)<1e-12)
 upper=np.isfinite(high)&~equality;lower=np.isfinite(low)&~equality
 matrix=sparse.vstack((C[equality],C[upper],-C[lower]),format='csc');rhs=np.r_[high[equality],high[upper],-low[lower]]
 cones=[clarabel.ZeroConeT(int(equality.sum())),clarabel.NonnegativeConeT(int(upper.sum()+lower.sum()))]
 settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=200;settings.tol_gap_abs=1e-8;settings.tol_gap_rel=1e-8;settings.tol_feas=1e-8
 result=clarabel.DefaultSolver(sparse.triu(H,format='csc'),np.asarray(g),matrix,rhs,cones,settings).solve();x=np.array(result.x);values=C@x
 violation=float(max(np.max(low-values),np.max(values-high),0.));success=str(result.status) in ('Solved','AlmostSolved') and np.isfinite(x).all() and violation<2e-5
 return SimpleNamespace(x=x,info=SimpleNamespace(status=str(result.status)+f'; constraint residual {violation:.3g}',status_val=1 if success else -1))
