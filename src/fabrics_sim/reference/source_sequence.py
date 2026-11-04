"""Select successive non-flight source swings without skipping support phases."""
import numpy as np


def next_source_landing(modes, after_frame, support):
    modes = np.asarray(modes)
    if modes.ndim != 1 or not np.isin(modes, [1, 2, 3]).all() or support not in (0, 1):
        raise ValueError('Expected non-flight contact modes and a support index')
    start = int(after_frame)
    if start != after_frame or not 0 <= start < len(modes):
        raise ValueError('Expected a valid integer source boundary')
    takeoff = start + 1
    while takeoff < len(modes) and modes[takeoff] == 3:
        takeoff += 1
    if takeoff == len(modes):
        raise ValueError('No next source swing exists')
    if modes[takeoff] != support + 1:
        raise ValueError('Next source swing disagrees with measured support')
    landing = takeoff
    while landing < len(modes) and modes[landing] == support + 1:
        landing += 1
    if landing == len(modes) or modes[landing] != 3:
        raise ValueError('Next source swing has no double-support landing')
    return takeoff, landing



def hold_takeoff_clock(clock, takeoff_frame):
    """A bounded transfer owns the clock from takeoff, before any swing advance."""
    from dataclasses import replace
    if clock.previous_frame > takeoff_frame+1e-6 or clock.frame < takeoff_frame-1e-6:
        raise RuntimeError('Transfer handoff must bracket its source takeoff')
    return replace(clock,frame=float(takeoff_frame),next_frame=float(takeoff_frame),rate=0.)


def source_recovery_active(completed_steps, after_step, recovery_steps=0):
    """A bounded recovery window; zero length preserves legacy indefinite recovery."""
    if (any(type(x) is not int or x < 0 for x in (completed_steps,after_step,recovery_steps))
            or (recovery_steps and (not after_step or recovery_steps % 2))):
        raise ValueError('Recovery needs nonnegative counts and an even bounded window after a source step')
    return bool(after_step and completed_steps >= after_step
                and (not recovery_steps or completed_steps < after_step+recovery_steps))


def audit_source_schedule(modes):
    """Describe the entire contact schedule without claiming physical feasibility."""
    modes=np.asarray(modes)
    if modes.ndim != 1 or not len(modes) or not np.isin(modes,[0,1,2,3]).all():
        raise ValueError('Expected a nonempty contact-mode vector containing 0,1,2,3')
    starts=np.r_[0,np.flatnonzero(np.diff(modes))+1]
    ends=np.r_[starts[1:],len(modes)]
    segments=[dict(start_frame=int(a),end_frame=int(b-1),mode=int(modes[a])) for a,b in zip(starts,ends)]
    singles=[s for s in segments if s['mode'] in (1,2)]
    repeated=[dict(previous_end_frame=a['end_frame'],next_takeoff_frame=b['start_frame'],mode=b['mode'])
              for a,b in zip(singles,singles[1:]) if a['mode']==b['mode']]
    direct=[dict(frame=b['start_frame'],previous_mode=a['mode'],next_mode=b['mode'])
            for a,b in zip(segments,segments[1:]) if a['mode'] in (1,2) and b['mode'] in (1,2)]
    flights=[s for s in segments if s['mode']==0]
    obstacles=([dict(frame=r['next_takeoff_frame'],reason='Repeated support requires a same-side swing') for r in repeated]
        +[dict(frame=r['frame'],reason='Direct support switch has no double-support landing') for r in direct]
        +[dict(frame=r['start_frame'],reason='Flight is unsupported by the stopped-step planner') for r in flights])
    if segments[-1]['mode']!=3:
        obstacles.append(dict(frame=segments[-1]['start_frame'],reason='No final double-support settling phase'))
    return dict(frame_count=len(modes),segment_count=len(segments),single_support_segments=len(singles),
                segments=segments,repeated_support=repeated,direct_switches=direct,flight_segments=flights,
                first_legacy_schedule_obstacle=min(obstacles,key=lambda r:r['frame']) if obstacles else None,
                legacy_alternating_schedule_compatible=not obstacles,
                final_double_support=segments[-1] if segments[-1]['mode']==3 else None,
                physical_feasibility_checked=False,full_tracking_passed=False)


def next_source_event(modes, after_frame):
    """Return the actual next support event or final two-foot tail, without skipping frames.

    A direct switch requires an explicitly inserted physical landing/transfer;
    the event table itself is not a dynamics feasibility certificate.
    """
    modes=np.asarray(modes)
    if modes.ndim!=1 or not len(modes) or not np.isin(modes,[1,2,3]).all():
        raise ValueError('Source events require non-flight contact modes')
    start=int(after_frame)
    if start!=after_frame or not 0<=start<len(modes):
        raise ValueError('Invalid source event boundary')
    takeoff=start
    if modes[start]!=3 and start>0 and modes[start-1]==modes[start]:
        raise ValueError('Source boundary lies inside an unfinished swing')
    while takeoff<len(modes) and modes[takeoff]==3:
        takeoff+=1
    if takeoff==len(modes):
        return dict(kind='tail',start_frame=start,end_frame=len(modes)-1,support=None,
                    takeoff_frame=None,inserted_double_support=False)
    support=int(modes[takeoff])-1
    landing=takeoff+1
    while landing<len(modes) and modes[landing]==support+1:
        landing+=1
    if landing==len(modes):
        raise ValueError('Source ends without a landing event')
    return dict(kind='swing',start_frame=start,end_frame=landing,takeoff_frame=takeoff,
                support=support,inserted_double_support=bool(modes[landing]!=3))
