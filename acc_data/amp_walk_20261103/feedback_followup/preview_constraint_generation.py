"""Exact row generation: accept only after checking every collision inequality."""
import time
import numpy as np
from scipy import sparse
import clarabel

def solve(P,g,A,b,cones,collision_A,collision_b,block_sizes,priorities,settings,initial_per_block=8,max_rounds=6,tolerance=1e-7):
 count=len(collision_b);selected=np.zeros(count,dtype=bool);offset=0
 for size in block_sizes:
  if size:
   local=np.argsort(priorities[offset:offset+size])[-min(initial_per_block,size):]
   selected[offset+local]=True
  offset+=size
 assert offset==count
 started=time.monotonic();history=[]
 for round_index in range(max_rounds+1):
  if round_index==max_rounds:selected[:]=True
  reduced_A=sparse.vstack([A,collision_A[selected]],format='csc');reduced_b=np.r_[b,collision_b[selected]]
  extra=[clarabel.NonnegativeConeT(int(selected.sum()))] if selected.any() else []
  solution=clarabel.DefaultSolver(P,g,reduced_A,reduced_b,cones+extra,settings).solve()
  x=np.asarray(solution.x);residual=collision_A@x-collision_b
  violation=float(np.max(residual,initial=0.));history.append(dict(rows=int(selected.sum()),full_collision_violation=violation,status=str(solution.status),iterations=int(getattr(solution,'iterations',0))))
  if str(solution.status) not in ('Solved','AlmostSolved'):
   if not selected.all():
    # A reduced problem can be poorly conditioned even when the full
    # formulation is solvable. Confirm failure with every original row.
    selected[:]=True
    continue
   break
  if violation<=tolerance or selected.all():break
  selected|=residual>tolerance
 return solution,dict(total_collision_rows=count,selected_collision_rows=int(selected.sum()),rounds=history,solve_seconds=time.monotonic()-started,full_collision_linear_violation=violation)
