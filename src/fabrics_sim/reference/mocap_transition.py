"""Whole-model load transfer and finite joins for managed PKL footsteps.

Only intermediate references change. Contact ownership remains with the measured
contact manager, and the source reference is restored before its landing frame.
"""
import torch
from .bridge import WholeBodyReferenceBridge
from .contact_transfer import ContactTransferReference, TransferStartGate
from .whole_body import WholeBodyReference
from .initial_stance import contact_pose_reference


class MocapContactTransition:
    def __init__(self, fabric, joint_names):
        self.fabric, self.joint_names = fabric, joint_names
        self.transfer = None
        self.entry = self.exit = None
        self.event = None
        self.release_time = None
        self.exit_duration = None
        self.phase = 'SOURCE'
        self.start_gate = TransferStartGate()

    def ready(self, time, com, velocity, q):
        return (self.transfer is not None and self.release_time is None
                and self.transfer.ready_for_release(time, com, velocity, q))

    def unload_weight(self, time):
        if self.transfer is None or self.release_time is not None:
            return 0.
        u = min(1., max(0., (time-self.transfer.start_time)/self.transfer.duration))
        return u**3*(10.-15.*u+6.*u*u)

    def sample(self, source, measured, feet, time, clock, manager, fps, stopped,
               com_velocity, normal_forces, dt):
        # A new transfer is owned by its source takeoff event, not by wall time.
        pending = clock.phase == 'PREPARE_STEP' and self.event != manager.pending_takeoff
        if pending:
            settled = self.start_gate.update(stopped, com_velocity, measured.v[:,6:], normal_forces, dt)
            if not settled:
                self.phase = 'WAIT_FOR_MEASURED_REST'
                return source
        else:
            self.start_gate.elapsed = 0.
        if pending:
            if self.exit is not None and time < self.release_time+self.exit_duration:
                raise RuntimeError('Next transfer starts before the previous source join completed')
            self.event = manager.pending_takeoff
            support = int(manager.modes[self.event])-1
            # Keep the stopped source torso/arms and COM. A low measured
            # velocity does not imply its posture is statically supportable.
            # Fit the leg configuration to actual anchors before constructing
            # the path; do not add a joint-space bridge that moves stance feet.
            com = self.fabric.dynamics_tasks.model.snapshot(source.q)['com']
            start_q = contact_pose_reference(self.fabric, source.q, self.joint_names,
                feet, com[:, :2], float(com[0, 2]), tolerance=2e-6)
            stationary = WholeBodyReference(start_q, torch.zeros_like(source.v),
                                             torch.zeros_like(source.a))
            self.transfer = ContactTransferReference(self.fabric, stationary, self.joint_names,
                feet.clone(), support, time, transfer_duration=2., preflight_dt=.02)
            self.entry = None
            self.exit = None
            self.release_time = None
            self.phase = 'TRANSFER'
        if self.transfer is None:
            return source
        if self.release_time is None and clock.mode == 3:
            reference = self.transfer.sample(time, 3)
            self.phase = self.transfer.phase
            return reference
        if self.release_time is None:
            if clock.mode != self.transfer.support+1 or manager.landing_frame is None:
                raise RuntimeError('Unexpected support or missing landing during source join')
            remaining = (manager.landing_frame-clock.frame)/fps/max(manager.speed,1e-8)
            if remaining <= .04:
                raise RuntimeError('Insufficient source time to join before original touchdown')
            self.exit_duration = min(.25, .5*remaining)
            self.release_time = time
            # Start at the measured state; never move/reset the physical robot.
            # Acceleration is the preceding WBC prediction, not a measurement.
            self.exit = WholeBodyReferenceBridge(measured, source, self.exit_duration)
        elapsed = time-self.release_time
        if elapsed < self.exit_duration and clock.mode == 3:
            raise RuntimeError('Touchdown began before the finite source join completed')
        self.phase = 'SOURCE_JOIN' if elapsed < self.exit_duration else 'SOURCE'
        return self.exit.sample(elapsed, source)
