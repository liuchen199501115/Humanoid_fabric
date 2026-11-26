"""Release unused feedback reserve only for torque control at every physics tick."""
import math

def effort_fraction(time,start,fraction,control_dt,physics_dt,*,direct_wbc,tracking_servo):
    if not all(math.isfinite(x) for x in (time,start,fraction,control_dt,physics_dt)):
        raise ValueError('Effort budget inputs must be finite')
    if not .8<=fraction<=1. or control_dt<=0 or physics_dt<=0:
        raise ValueError('Effort budget must stay within the physical motor limit')
    if time<start:return .8
    if not direct_wbc or not tracking_servo or not math.isclose(control_dt,physics_dt,rel_tol=1e-9,abs_tol=1e-12):
        raise ValueError('Reserve release requires direct WBC with measured servo seeds every physics tick')
    phase=min(1.,max(0.,(time-start)/.06));blend=phase**3*(10-15*phase+6*phase**2)
    return .8+(fraction-.8)*blend
