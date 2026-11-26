"""Independent interior-point solve of the saved offline reference QP."""
from pathlib import Path
import json,sys,time
import numpy as np
from scipy import sparse
import clarabel

def solve(H,linear,C,low,high):
    row_scale=np.maximum(np.sqrt(C.multiply(C).sum(axis=1)).A.ravel(),1e-8)
    scale=max(float(H.diagonal().max()),1.)
    Cs=(sparse.diags(1/row_scale)@C).tocsc();lo=np.asarray(low)/row_scale;hi=np.asarray(high)/row_scale
    equality=np.isfinite(lo)&np.isfinite(hi)&(lo==hi)
    upper=np.isfinite(hi)&~equality;lower=np.isfinite(lo)&~equality
    A=sparse.vstack((Cs[equality],Cs[upper],-Cs[lower]),format='csc');b=np.r_[hi[equality],hi[upper],-lo[lower]]
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1;settings.max_iter=200;settings.tol_feas=1e-9;settings.tol_gap_abs=1e-9;settings.tol_gap_rel=1e-9
    answer=clarabel.DefaultSolver(sparse.triu(H/scale,format='csc'),linear/scale,A,b,[clarabel.ZeroConeT(int(equality.sum())),clarabel.NonnegativeConeT(int(upper.sum()+lower.sum()))],settings).solve()
    x=np.asarray(answer.x);residual=max(0.,float(np.max(np.asarray(low)-C@x)),float(np.max(C@x-np.asarray(high))))
    return x,dict(status=str(answer.status),constraint_residual=residual,iterations=answer.iterations,accepted=str(answer.status)=='Solved' and residual<1e-5)

if __name__=='__main__':
    p=Path(__file__).parent;name=sys.argv[1] if len(sys.argv)>1 else 'wbc_turn_upper_centroidal_fit';d=np.load(p/(name+'.problem.npz'))
    H=sparse.csc_matrix((d['H_data'],d['H_indices'],d['H_indptr']),shape=d['H_shape']);C=sparse.csc_matrix((d['C_data'],d['C_indices'],d['C_indptr']),shape=d['C_shape'])
    started=time.monotonic();x,report=solve(H,d['linear'],C,d['low'],d['high']);report['wall_seconds']=time.monotonic()-started;print(report,flush=True)
    (p/(name+'.solver_audit.json')).write_text(json.dumps(report,indent=2)+'\n')
    if report['accepted']:np.savez_compressed(p/(name+'.solution.npz'),x=x)
    else:raise RuntimeError('Offline QP did not pass exact constraint validation')
