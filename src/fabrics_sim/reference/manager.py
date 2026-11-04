"""Flat-ground walking prototype: one clock and one contact-mode owner."""
from dataclasses import dataclass
import math
import numpy as np


def contact_sequence_modes(sequence, frame_count):
    """Preserve supplied left/right labels; this manager cannot model flight."""
    sequence = np.asarray(sequence)
    if (sequence.shape != (frame_count, 2) or frame_count < 1
            or not np.isin(sequence, [0, 1]).all()):
        raise ValueError('Expected binary contact_sequence [frames,2] in left/right order')
    modes = sequence[:, 0].astype(np.int64) + 2*sequence[:, 1].astype(np.int64)
    if (modes == 0).any():
        raise ValueError('ReferenceManager does not support flight in contact_sequence')
    return modes


def infer_corner_contact_plan(points, fps, height_threshold=.025, speed_threshold=.20):
    """A near-ground slow sole point can support a rolling, moving ankle.

    Height and speed must belong to the SAME point. These remain inferred
    candidates, not measured contacts or a guarantee of flat-foot support.
    """
    points=np.asarray(points,dtype=float)
    if (points.ndim!=4 or points.shape[1]!=2 or points.shape[-1]!=3 or not points.shape[2]
            or not len(points) or not np.isfinite(points).all() or not math.isfinite(fps) or fps<=0):
        raise ValueError('Expected finite [frames,2,corners,3] and positive FPS')
    velocity=np.gradient(points,1./fps,axis=0) if len(points)>1 else np.zeros_like(points)
    speed=np.linalg.norm(velocity,axis=-1)
    height=points[...,2]
    supported_speed=np.where(height<=height_threshold,speed,1e6).min(-1)
    modes,ambiguous=infer_contact_plan(height.min(-1),supported_speed,fps,height_threshold,speed_threshold)
    return modes,ambiguous,supported_speed


def infer_contact_plan(sole_height, foot_speed, fps, height_threshold=.025, speed_threshold=.20):
    """Candidate labels, not ground truth; preserve ambiguous-frame flags."""
    height = np.asarray(sole_height)
    speed = np.asarray(foot_speed)
    if height.shape != speed.shape or height.ndim != 2 or height.shape[1] != 2:
        raise ValueError('Expected [T,2] heights/speeds')
    if len(height)==0 or fps <= 0 or not np.isfinite(height).all() or not np.isfinite(speed).all():
        raise ValueError('Invalid contact-plan data')
    candidate = (height <= height_threshold) & (speed <= speed_threshold)
    ambiguous = ~candidate.any(1)
    # Non-flight prototype: retain the lower foot, but explicitly flag uncertainty.
    candidate[ambiguous, height[ambiguous].argmin(1)] = True
    confirmed = candidate[0].copy(); counts = np.zeros(2, dtype=int)
    result=[]; confirmation=max(2, round(.06*fps))
    for row in candidate:
        counts=np.where(row != confirmed, counts+1, 0)
        change=counts>=confirmation
        confirmed=np.where(change,row,confirmed)
        counts[change]=0
        mode=int(confirmed[0])+2*int(confirmed[1])
        result.append(mode or int(np.argmin(height[len(result)]))+1)
    return np.asarray(result,dtype=np.int64),ambiguous


@dataclass(frozen=True)
class ManagedClock:
    frame: float
    previous_frame: float
    next_frame: float
    mode: int
    support_mode: int
    phase: str
    rate: float
    waiting_seconds: float


class ReferenceManager:
    """Scalar scheduler for one robot; no physics state resets or torque output."""
    def __init__(self, modes, fps, start_frame=0, speed=1., ramp_duration=1., timeout=6., min_load=100., settle_duration=.3,
                 early_touchdown=False, settle_on_incoming=False, align_takeoff_frame=False,
                 continuous_source_clock=False, takeoff_lead=0.):
        self.modes=np.asarray(modes,dtype=np.int64)
        if self.modes.ndim!=1 or not len(self.modes) or not np.isin(self.modes,[1,2,3]).all():
            raise ValueError('Prototype supports only non-flight walking modes 1,2,3')
        if (not all(math.isfinite(v) for v in (fps,start_frame,speed,ramp_duration,timeout,min_load,settle_duration))
                or not 0 <= start_frame < len(self.modes) or min(fps,speed,ramp_duration,timeout,min_load,settle_duration)<=0):
            raise ValueError('Invalid reference-manager configuration')
        self.fps=float(fps); self.speed=float(speed); self.tau=float(start_frame)/fps
        self.rate=0.; self.max_rate_change=speed/ramp_duration
        self.search_from = int(start_frame)
        self.settling = False
        self.settle_time = 0.
        self.settle_duration = settle_duration
        self.early_touchdown = bool(early_touchdown)
        self.settle_on_incoming = bool(settle_on_incoming)
        self.align_takeoff_frame = bool(align_takeoff_frame)
        if not math.isfinite(takeoff_lead) or not 0. <= takeoff_lead <= .15:
            raise ValueError('Takeoff lead must be between 0 and .15 source seconds')
        if takeoff_lead and not align_takeoff_frame:
            raise ValueError('Takeoff lead requires aligned takeoff')
        self.takeoff_lead=float(takeoff_lead)
        self.continuous_source_clock = bool(continuous_source_clock)
        self.landing_support = 3
        self.swing_clock = None
        self.swing_elapsed = 0.
        self.last_dt = None
        self.touchdown_confirmed = 0
        self.transfer_load_latched = False
        self.mode=3; self.phase='INITIALIZE'; self.wait=0.; self.confirmed=0
        self.timeout=timeout; self.min_load=min_load; self.swing_clear=False; self.takeoff_frame=0; self.pending_takeoff=None; self.landing_frame=None

    def synchronize_swing(self,duration):
        """Finish this source phase with its timed foot/COM trajectory."""
        from .phase_clock import monotone_swing_clock
        if self.mode not in (1,2) or self.landing_frame is None:
            raise ValueError('Timed swing requires a planned single-support endpoint')
        self.swing_clock=monotone_swing_clock(self.tau,self.rate,self.landing_frame/self.fps,
                                              duration,self.landing_rate_limit(self.landing_frame))
        self.swing_elapsed=0.

    def landing_rate_limit(self,frame):
        """Source rate allowed by stopping room after a proposed landing."""
        if not 0<=frame<len(self.modes):raise ValueError('Landing frame outside source sequence')
        # Average swing rate is not the landing rate. After a stretched swing
        # the body can still enter double support at nominal source speed.
        # Check stopping room and polynomial monotonicity below instead of
        # forcing the whole next transfer to inherit the average swing speed.
        end_rate=self.speed if self.continuous_source_clock else 0.
        if self.continuous_source_clock:
            upcoming=np.flatnonzero(self.modes[frame:] != 3)
            if len(upcoming):
                # The next takeoff is not yet authorized. Retain enough
                # source-time stopping room after this landing, including
                # the preparation margin and one discrete control update.
                room=max(0.,int(upcoming[0])/self.fps-.05)
                dt=0. if self.last_dt is None else self.last_dt
                braking=self.max_rate_change
                cap=max(0.,math.sqrt((braking*dt)**2+2*braking*room)-braking*dt)
                end_rate=min(end_rate,cap)
        return end_rate

    def outgoing_foot(self, support_mode):
        """Old support during loaded transfer, including SETTLE -> TRACK.

        Foot index is zero-based. A transient TRACK phase does not change
        ownership of the outgoing contact or reset its load confirmation.
        """
        if self.mode != 3:
            return None
        if self.phase == 'PREPARE_STEP' and support_mode in (1,2):
            return 2-support_mode
        return self.landing_support-1 if self.landing_support in (1,2) else None

    def transfer_load_ready(self, forces, support_mode):
        """Necessary measured-load gate shared by scheduling and preview work."""
        forces = np.asarray(forces, dtype=float)
        if (forces.shape != (2,) or not np.isfinite(forces).all()
                or support_mode not in (1,2)):
            raise ValueError('Load transfer needs two finite forces and a support foot')
        total = forces.clip(0).sum()
        index = support_mode-1
        # Enter only after sustained 80% loading; after confirmation retain
        # readiness down to the same 65% used for the measured toe transfer.
        # A small contact redistribution must not revoke a step precisely at
        # the takeoff frame after the clock has committed its approach.
        fraction = .65 if self.transfer_load_latched else .8
        return bool(total >= self.min_load and forces[index] >= fraction*total
                    and forces[1-index] <= (1.-fraction)*total+1e-9)

    def step(self, forces, measured_contacts, dt, warming_up=False, sole_clearance=None, balance_ready=None,
             sole_max_height=None, touchdown_contact=None):
        forces=np.asarray(forces,dtype=float); measured=np.asarray(measured_contacts,dtype=bool)
        if forces.shape!=(2,) or measured.shape!=(2,) or not np.isfinite(forces).all() or not math.isfinite(dt) or dt<=0:
            raise ValueError('Invalid observation')
        self.last_dt=float(dt)
        old=self.tau; old_rate=self.rate
        confirmation=max(1,math.ceil(.06/dt-1e-9))
        frame=min(int(round(self.tau*self.fps)),len(self.modes)-1)
        total=forces.clip(0).sum()
        if touchdown_contact is not None:
            touchdown_contact=np.asarray(touchdown_contact,dtype=bool)
            if touchdown_contact.shape!=(2,):
                raise ValueError('Expected two observed touchdown-patch flags')
        if sole_max_height is not None:
            sole_max_height=np.asarray(sole_max_height,dtype=float)
            if sole_max_height.shape!=(2,) or not np.isfinite(sole_max_height).all():
                raise ValueError('Expected finite maximum sole heights [2]')
        if self.early_touchdown and (sole_clearance is None or sole_max_height is None):
            raise ValueError('Early touchdown requires minimum and maximum sole heights')
        if sole_clearance is not None:
            sole_clearance=np.asarray(sole_clearance,dtype=float)
            if sole_clearance.shape!=(2,) or not np.isfinite(sole_clearance).all():
                raise ValueError('Expected finite sole clearances [2]')
            # A flat unloaded foot may still touch the floor. Permit it only
            # in double support with measured load elsewhere; single support
            # always requires the load detector. Wrench bounds still allow zero.
            if self.mode==3 and total>=self.min_load:
                measured=measured | (sole_clearance<=.002)
        support=self.mode; target_rate=self.speed
        if warming_up:
            self.phase='INITIALIZE'; self.rate=0.
            return ManagedClock(self.tau*self.fps,self.tau*self.fps,self.tau*self.fps,self.mode,self.mode,self.phase,0.,0.)
        # A non-existent physical contact is never retained as a virtual support.
        observed=int(measured[0])+2*int(measured[1])
        observed_takeoff=False
        if self.mode & observed != self.mode:
            # During continuous walking the confirmed outgoing foot can leave
            # before the nominal source-time gate. If the incoming sole has
            # finished settling, sustained load is already confirmed and the
            # CURRENT preview approves this same next support, hand contact
            # ownership to it now. Never invent the missing old contact, skip
            # settling, jump source frames, or waive the live feasibility gate.
            next_support=(None if self.pending_takeoff is None else
                          int(self.modes[self.pending_takeoff]))
            observed_takeoff=bool(self.continuous_source_clock and self.mode==3
                and not self.settling and self.transfer_load_latched
                and self.confirmed>=confirmation and observed==next_support
                and balance_ready is not None and bool(balance_ready[observed-1])
                and self.transfer_load_ready(forces,observed))
            if not observed_takeoff:
                raise RuntimeError(f'ReferenceManager lost support: expected {self.mode}, observed {observed}')
        search_start = max(frame, self.search_from)
        future=np.flatnonzero((self.modes[search_start:] != 3) & (self.modes[search_start:] != self.mode))
        event=search_start+int(future[0]) if len(future) else None
        distance=(event/self.fps-self.tau) if event is not None else math.inf
        horizon=max(.4, self.rate**2/(2*self.max_rate_change)+.05+2*dt*self.rate)
        if self.mode==3 and event is not None and distance<=horizon:
            self.pending_takeoff = event if self.pending_takeoff is None else self.pending_takeoff
        preparing=self.mode==3 and self.pending_takeoff is not None
        if preparing:
            next_support=int(self.modes[self.pending_takeoff]); idx=next_support-1
            ready=measured[idx] and self.transfer_load_ready(forces,next_support)
            if balance_ready is not None:
                ready=ready and bool(balance_ready[idx])
            # SETTLE and PREPARE_STEP observe the same incoming foot. Count
            # uninterrupted load/preview readiness through that phase change;
            # release still waits for completed settling and the source gate.
            self.confirmed=self.confirmed+1 if ready else 0
            if not ready:
                self.transfer_load_latched=False
            elif self.confirmed>=confirmation:
                self.transfer_load_latched=True
        if self.settling:
            self.phase='SETTLE';support=self.landing_support;target_rate=0.
            if self.continuous_source_clock:
                # Load confirmation owns contact release, not playback. Keep
                # source speed when there is room, while retaining the same
                # braking margin before the next unconfirmed takeoff.
                room=max(0.,distance-.05)
                target_rate=min(self.speed,max(0.,math.sqrt(
                    (self.max_rate_change*dt)**2+2*self.max_rate_change*room)
                    -self.max_rate_change*dt))
            self.wait+=dt
            settled_load = (forces >= 20.).all()
            if self.settle_on_incoming:
                # In continuous walking the trailing foot is allowed to
                # unload. Requiring it to reload contradicts the capture
                # controller's transfer toward the newly landed foot.
                incoming = 2-self.landing_support
                settled_load = (total >= self.min_load and measured[incoming]
                                and forces[incoming] >= max(20., .5*total))
                if sole_max_height is not None:
                    settled_load = settled_load and sole_max_height[incoming] <= .002
            self.settle_time=self.settle_time+dt if settled_load else 0.
            if self.settle_time>=self.settle_duration:
                self.settling=False;self.wait=0.;self.phase='TRACK';support=3
        elif preparing:
            event=self.pending_takeoff
            distance=event/self.fps-self.tau
            support=int(self.modes[event]); idx=support-1
            self.phase='PREPARE_STEP'; self.wait+=dt
            approach_rate=min(self.speed,self.rate+self.max_rate_change*dt)
            reaches_takeoff=self.tau+.5*(self.rate+approach_rate)*dt>=event/self.fps-self.takeoff_lead-1e-9
            if self.confirmed>=confirmation and (not self.align_takeoff_frame or reaches_takeoff or observed_takeoff):
                self.mode=support; self.phase='SWING'; self.wait=0.; self.confirmed=0; self.swing_clear=False; self.takeoff_frame=event
                self.transfer_load_latched=False
                self.touchdown_confirmed=0
                future_land=np.flatnonzero(self.modes[event:] != self.mode)
                self.landing_frame=event+int(future_land[0]) if len(future_land) else None
                self.pending_takeoff=None
            elif self.confirmed>=confirmation and self.align_takeoff_frame:
                # Balance readiness can precede the PKL takeoff. Continue its
                # preparation frames with both contacts until the event,
                # rather than swinging while the posture is still in stance.
                target_rate=self.speed
            else:
                # Brake ahead of the event; no instantaneous reference freeze.
                distance=max(0.,distance-.05)
                target_rate=min(target_rate,max(0., math.sqrt((self.max_rate_change*dt)**2+2*self.max_rate_change*distance)-self.max_rate_change*dt))
        elif self.mode in (1,2):
            self.phase='SWING'; support=self.mode; swing=1 if self.mode==1 else 0
            self.swing_clear |= ((not measured[swing] or forces[swing]<10.)
                                 if sole_clearance is None else sole_clearance[swing]>=.01)
            landing_distance=math.inf if self.landing_frame is None else self.landing_frame/self.fps-self.tau
            wants_land=landing_distance <= horizon
            immediate=False
            if sole_clearance is None:
                landing=self.swing_clear and measured[swing] and forces[swing]>=20.
            else:
                # Initial touchdown is not load-transfer readiness: a foot on
                # the floor can carry only a few newtons while COM stays over
                # the old support. Require geometry, force and three samples.
                touching=self.swing_clear and sole_clearance[swing]<=.002 and forces[swing]>=5.
                self.touchdown_confirmed=self.touchdown_confirmed+1 if touching else 0
                # A loaded, geometrically flat foot already affects physical
                # dynamics. Do not continue driving it as a free swing during
                # load debounce. SETTLE still confirms load before the next step.
                flat = sole_max_height is None or sole_max_height[swing] <= .002
                # At 50 Hz a settled flat sole can initially carry only its
                # own small load (about 2 N), then lift again before reaching
                # the 5 N debounce threshold. Geometry plus a positive 1 N
                # load establishes contact; SETTLE still confirms bearing.
                immediate = (self.early_touchdown and self.swing_clear and flat
                             and sole_clearance[swing] <= .002 and forces[swing] >= 1.)
                if self.early_touchdown and touchdown_contact is not None:
                    # Caller supplies a real loaded sole patch and models its
                    # own point/line constraints. A heel need not wait until
                    # the full sole is flat to enter the dynamics model.
                    immediate=immediate or (touching and touchdown_contact[swing])
                # Debouncing force cannot turn an edge contact into a flat
                # sole. This manager activates six rigid contact coordinates;
                # require the supplied full-sole geometry on both paths.
                landing = immediate or (self.touchdown_confirmed >= confirmation and flat)
            if landing and (immediate or wants_land or forces[swing]>.25*max(total,1.)):
                # Touchdown can precede its source frame: do not replay the
                # completed swing when the source clock is still braking.
                if self.landing_frame is not None:
                    self.search_from = self.landing_frame
                self.landing_support=self.mode
                self.settling=True;self.settle_time=0.
                self.mode=3; support=3; self.phase='LANDING'; self.wait=0.
                target_rate=min(target_rate,.5*self.speed)
            if wants_land and not landing:
                self.phase='LANDING'; self.wait+=dt
                distance=max(0.,landing_distance-.05)
                target_rate=min(target_rate,max(0.,math.sqrt((self.max_rate_change*dt)**2+2*self.max_rate_change*distance)-self.max_rate_change*dt))
            else:
                self.wait=0.
        else:
            self.phase='TRACK'; self.wait=0.
        if self.wait>self.timeout:
            raise RuntimeError(f'ReferenceManager {self.phase} timed out after {self.wait:.2f}s at frame {frame}; progress is not success')
        if self.swing_clock is not None:
            self.swing_elapsed+=dt
            self.tau,self.rate,acceleration=self.swing_clock.sample(self.swing_elapsed)
            next_tau=self.swing_clock.sample(self.swing_elapsed+dt)[0]
            if self.mode==3 and self.swing_elapsed>=self.swing_clock.duration:
                if self.continuous_source_clock:
                    overshoot=self.swing_elapsed-self.swing_clock.duration
                    if overshoot>=dt-1e-9:
                        # Late measured touchdown resumes from its held state;
                        # never catch up source frames accumulated while waiting.
                        self.rate=old_rate
                        overshoot=0.
                    self.tau=min(self.tau+overshoot*self.rate,
                                 (len(self.modes)-1)/self.fps)
                    next_tau=min(self.tau+dt*self.rate,(len(self.modes)-1)/self.fps)
                self.swing_clock=None
            elif self.swing_elapsed>=self.swing_clock.duration:
                # An unconfirmed touchdown cannot advance past the endpoint,
                # even when continuous walking requested a nonzero end rate.
                self.rate=0.
        else:
            change=np.clip(target_rate-self.rate,-self.max_rate_change*dt,self.max_rate_change*dt)
            self.rate=max(0.,self.rate+float(change))
            self.tau=min(self.tau+dt*(old_rate+self.rate)*.5,(len(self.modes)-1)/self.fps)
            acceleration=(self.rate-old_rate)/dt
            next_tau=min(max(self.tau,self.tau+self.rate*dt+.5*acceleration*dt*dt),(len(self.modes)-1)/self.fps)
            # An authorized takeoff changes mode on the tick that crosses its
            # source frame. Clamping tau/next_tau to the event and resuming on
            # the next tick creates a false ~300/s clock acceleration at 500 Hz.
            # The boundary check below still rejects any UNAUTHORIZED crossing.
        boundary = self.pending_takeoff if self.mode==3 else self.landing_frame
        if boundary is not None and self.tau*self.fps > boundary + 1e-6:
            raise RuntimeError(f'ReferenceManager cannot stop before unconfirmed contact event at frame {boundary}')
        # Symmetric samples encode the same clock's speed AND acceleration.
        return ManagedClock(self.tau*self.fps,old*self.fps,next_tau*self.fps,self.mode,support,self.phase,self.rate,self.wait)
