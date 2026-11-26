"""Seed near-future support from observed points until a foot leaves stance."""
def hold_observed_patches(patches,frames,elapsed,initial,planned_mode,seconds):
 if seconds<=0:return
 for side,indices in initial.items():
  side=int(side);continuous=True
  for node in range(1,len(frames)):
   continuous=continuous and bool(planned_mode(frames[node])&(1<<side))
   if continuous and elapsed[node]<=seconds+1e-10:
    patches[node][side]=set(indices)
