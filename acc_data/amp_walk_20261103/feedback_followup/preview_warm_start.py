"""Shift a verified prediction in physical coordinates for the next SQP seed."""
import numpy as np

def shift_prediction(previous, reference, frame, rate, dt, horizon, integrate, difference, integration='continuous'):
    old_frames = previous['frames']
    if frame < old_frames[0] or frame >= old_frames[-1] or rate <= 0:
        return None
    intervals=np.broadcast_to(np.asarray(dt,dtype=float),(horizon,))
    if np.any(intervals<=0):raise ValueError('Prediction intervals must be positive')
    target_offsets=np.r_[0.,np.cumsum(intervals)]
    states, controls = [], []
    point_forces = {}
    for index in range(horizon + 1):
        target_frame = frame + rate * target_offsets[index]
        stage = min(max(int(np.searchsorted(old_frames, target_frame+1e-8, side='right') - 1), 0), len(old_frames) - 1)
        rq, rv, _ = [x.numpy().ravel() for x in reference.kinematics(float(old_frames[stage]), rate)]
        q = integrate(rq, previous['states'][stage, :35])
        v = rv + previous['states'][stage, 35:]
        control = previous['controls'][min(stage, len(previous['controls']) - 1)]
        elapsed = max(0.,(target_frame - old_frames[stage]) / rate)
        # At an impact node, its stored velocity is already post-impact.
        # Inside the preceding interval, integrate its pre-impact velocity;
        # interpolating node velocities would spread the impulse over time.
        if integration=='semiimplicit':
            from preview_integration import advance
            q,v=advance(q,v,control[:35],elapsed)
        elif integration=='continuous':
            q = integrate(q, elapsed * v + .5 * elapsed**2 * control[:35])
            v = v + elapsed * control[:35]
        else:raise ValueError('Unknown prediction integration method')
        nrq, nrv, _ = [x.numpy().ravel() for x in reference.kinematics(float(target_frame), rate)]
        states.append(np.r_[difference(q, nrq), v - nrv])
        if index < horizon:
            controls.append(control.copy())
            old_forces=previous.get('point_forces',{}).get(min(stage,len(previous['controls'])-1))
            if old_forces is not None:point_forces[index]=old_forces
    return dict(states=np.asarray(states), controls=np.asarray(controls),point_forces=point_forces)
