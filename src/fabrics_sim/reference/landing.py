"""Independent wall-time landing target; never driven by a frozen PKL clock."""
import numpy as np
from scipy.spatial.transform import Rotation


def contact_search(elapsed, depth, duration):
    """Bounded downward reference search; contact must be measured separately."""
    if not np.isfinite([elapsed,depth,duration]).all() or depth<0 or duration<=0:
        raise ValueError('Invalid contact-search boundary conditions')
    s=np.clip(elapsed/duration,0.,1.)
    return (-depth*(10*s**3-15*s**4+6*s**5),
            -depth*(30*s**2-60*s**3+30*s**4)/duration,
            -depth*(60*s-180*s**2+120*s**3)/duration**2)


def minimum_swing_duration(start_position, initial_velocity, goal_position,
                           maximum_speed, clearance=.06, minimum=.5):
    """Shortest speed-bounded duration for the actual translation polynomial.

    Evaluate all stationary points of squared speed, including the lift bump.
    The old triangle bound remains the feasible upper bracket. This checks
    Cartesian speed only, not joint limits or dynamic feasibility.
    """
    start,velocity,goal=(np.asarray(x,dtype=float) for x in
                         (start_position,initial_velocity,goal_position))
    if (any(x.shape!=(3,) for x in (start,velocity,goal))
            or not np.isfinite([start,velocity,goal]).all()
            or not np.isfinite([maximum_speed,clearance,minimum]).all()
            or maximum_speed<=0 or clearance<0 or minimum<=0):
        raise ValueError('Invalid swing speed boundary conditions')
    available=maximum_speed-np.linalg.norm(velocity)
    if available<=0:
        raise RuntimeError('Swing foot must slow before a speed-bounded step can start')
    def peak_speed(duration):
        d=(goal-start)/duration
        coefficients=np.stack((velocity,np.zeros(3),30*d-18*velocity,
                               -60*d+32*velocity,30*d-15*velocity,np.zeros(3)))
        coefficients[:,2]+=clearance/duration*np.array([0.,0.,192.,-768.,960.,-384.])
        squared=np.zeros(11)
        for c in coefficients.T:
            product=np.polynomial.polynomial.polymul(c,c)
            squared[:len(product)]+=product
        roots=np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(squared))
        candidates=[0.,1.]+[r.real for r in roots if abs(r.imag)<1e-7 and 0.<r.real<1.]
        # Evaluate the vector itself to avoid cancellation in squared speed.
        velocities=np.polynomial.polynomial.polyval(np.asarray(candidates),coefficients)
        return float(np.linalg.norm(velocities,axis=0).max())
    low=float(minimum)
    if peak_speed(low)<=maximum_speed:
        return low
    high=max(low,(1.875*np.linalg.norm(goal-start)+4*clearance)/available)
    # For each phase the feasible inverse-duration interval contains zero:
    # the initial-velocity basis has magnitude <= 1. Their intersection is
    # an interval too, so this bisection cannot skip another feasible branch.
    for _ in range(35):
        middle=.5*(low+high)
        if peak_speed(middle)<=maximum_speed*(1.-1e-10): high=middle
        else: low=middle
    return high


class LandingTransition:
    def __init__(self, start_time, start_pose, start_velocity, goal_pose, duration=2.,
                 clearance=0., search_depth=0., search_duration=1.):
        start_pose=np.asarray(start_pose,dtype=float);goal_pose=np.asarray(goal_pose,dtype=float)
        velocity=np.asarray(start_velocity,dtype=float)
        if start_pose.shape!=(7,) or goal_pose.shape!=(7,) or velocity.shape!=(6,) or duration<=0:
            raise ValueError('Landing needs poses [7], twist [6], positive duration')
        if not all(np.isfinite(x).all() for x in (start_pose,goal_pose,velocity,start_time,duration,
                                                  clearance,search_depth,search_duration)):
            raise ValueError('Landing inputs must be finite')
        if clearance < 0 or search_depth < 0 or search_duration <= 0:
            raise ValueError('Invalid clearance/contact-search configuration')
        self.clearance = float(clearance)
        self.search_depth = float(search_depth)
        self.search_duration = float(search_duration)
        self.start_time=float(start_time);self.duration=float(duration)
        self.position=start_pose[:3].copy();self.velocity=velocity[:3].copy();self.goal=goal_pose.copy()
        t=duration
        matrix=np.array([[t**3,t**4,t**5],[3*t*t,4*t**3,5*t**4],[6*t,12*t*t,20*t**3]])
        self.coefficients=np.linalg.solve(matrix,np.stack((goal_pose[:3]-self.position-self.velocity*t,-self.velocity,np.zeros(3))))
        self.rotation=Rotation.from_quat(start_pose[3:])
        self.goal[3:]=Rotation.from_quat(goal_pose[3:]).as_quat()
        # Polynomial in rotation-vector coordinates; match initial world angular velocity.
        self.rotation_goal=(self.rotation.inv()*Rotation.from_quat(goal_pose[3:])).as_rotvec()
        self.local_omega=self.rotation.inv().apply(velocity[3:])
        self.angular_coefficients=np.linalg.solve(matrix,np.stack((self.rotation_goal-self.local_omega*t,-self.local_omega,np.zeros(3))))

    def sample(self,time):
        t=np.clip(time-self.start_time,0.,self.duration)
        def polynomial(start,velocity,coeff):
            c3,c4,c5=coeff
            return (start+velocity*t+c3*t**3+c4*t**4+c5*t**5,
                    velocity+3*c3*t*t+4*c4*t**3+5*c5*t**4,
                    6*c3*t+12*c4*t*t+20*c5*t**3)
        p,v,a=polynomial(self.position,self.velocity,self.coefficients)
        s=t/self.duration
        # Zero height, velocity and acceleration at both ends; peak at mid-swing.
        p[2]+=64*self.clearance*(s**3-3*s**4+3*s**5-s**6)
        v[2]+=64*self.clearance*(3*s**2-12*s**3+15*s**4-6*s**5)/self.duration
        a[2]+=64*self.clearance*(6*s-36*s**2+60*s**3-30*s**4)/self.duration**2
        phi,_,_=polynomial(np.zeros(3),self.local_omega,self.angular_coefficients)
        quat=(self.rotation*Rotation.from_rotvec(phi)).as_quat()
        # Rotation vectors are nonlinear: differentiate SO(3), not its components.
        eps=1e-4
        def orientation(at):
            c3,c4,c5=self.angular_coefficients
            x=self.local_omega*at+c3*at**3+c4*at**4+c5*at**5
            return self.rotation*Rotation.from_rotvec(x)
        def omega(at):
            return (orientation(at+eps)*orientation(at-eps).inv()).as_rotvec()/(2*eps)
        w=omega(t);alpha=(omega(t+eps)-omega(t-eps))/(2*eps)
        if time-self.start_time>=self.duration:
            p=self.goal[:3].copy();quat=self.goal[3:].copy()
            v=np.zeros(3);a=np.zeros(3);w=np.zeros(3);alpha=np.zeros(3)
            # A bounded reference search, terminated by the caller on measured
            # contact. Never teleport the robot or assume the target is contact.
            offset,v[2],a[2]=contact_search(time-self.start_time-self.duration,
                                          self.search_depth,self.search_duration)
            p[2]+=offset
        return np.r_[p,quat],np.r_[v,w],np.r_[a,alpha]
