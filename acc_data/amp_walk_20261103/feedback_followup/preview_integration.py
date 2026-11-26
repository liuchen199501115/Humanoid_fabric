"""Match the simulator's 2 ms semi-implicit position update."""
import numpy as np
from preview_manifold import integrate
SIMULATION_DT=.002

def substeps(duration,step=SIMULATION_DT):
 if duration<0 or step<=0:raise ValueError('Nonnegative duration and positive step required')
 count=int(np.floor((duration+1e-12)/step));remainder=max(0.,duration-count*step)
 return [step]*count+([remainder] if remainder>1e-12 else [])
def position_factor(duration,step=SIMULATION_DT):
 steps=substeps(duration,step)
 return .5*(duration*duration+sum(h*h for h in steps))
def advance(q,v,a,duration,step=SIMULATION_DT):
 q=np.asarray(q).copy();v=np.asarray(v).copy()
 for h in substeps(duration,step):
  v=v+h*a;q=integrate(q,h*v)
 return q,v
