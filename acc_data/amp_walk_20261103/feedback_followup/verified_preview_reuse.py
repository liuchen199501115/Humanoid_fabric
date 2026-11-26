"""Reuse only a nearby, unexpired, previously verified prediction."""
import numpy as np
from preview_manifold import integrate,difference

def reusable_control(previous,q,v,frame,rate,mode):
 if previous is None or not previous.get('success') or not previous.get('nonlinear_verified'):return None
 if not all(k in previous for k in ('physical_q','physical_v','contact_modes')):return None
 frames=previous['frames']
 if rate<=0 or frame<frames[0] or frame>=frames[-1] or (frame-frames[0])/rate>.06:return None
 stage=min(int(np.searchsorted(frames,frame+1e-8,side='right')-1),len(previous['controls'])-1)
 if int(previous['contact_modes'][stage])!=int(mode):return None
 elapsed=max(0.,(frame-frames[stage])/rate);control=previous['controls'][stage]
 if previous.get('integration')=='semiimplicit':
  from preview_integration import advance
  qn,vn=advance(previous['physical_q'][stage],previous['physical_v'][stage],control[:35],elapsed)
 else:
  qn=integrate(previous['physical_q'][stage],elapsed*previous['physical_v'][stage]+.5*elapsed**2*control[:35])
  vn=previous['physical_v'][stage]+elapsed*control[:35]
 error=difference(q,qn);dv=v-vn
 if np.linalg.norm(error[:3])>.025 or np.linalg.norm(error[3:6])>.07 or np.max(abs(error[6:]))>.1:return None
 if np.linalg.norm(dv[:3])>.5 or np.linalg.norm(dv[3:6])>1. or np.max(abs(dv[6:]))>3.:return None
 # Only the first stored wrench uses the original WBC contact coordinates.
 wrench=control[35:47].copy() if stage==0 else None
 return dict(plan=(qn,vn,control[:35].copy(),wrench),stage=stage,age_seconds=float((frame-frames[0])/rate))
