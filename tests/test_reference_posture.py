import unittest
import torch
from fabrics_sim.reference.posture import crouched_posture, crouched_posture_kinematics


class CrouchedPostureTests(unittest.TestCase):
    names=[s+'_'+joint+'_joint' for s in ('left','right')
           for joint in ('hip_pitch','knee','ankle_pitch')]+['arm_joint']

    def test_preserves_pitch_sum_and_upper_body(self):
        q=torch.tensor([[.1,-.03,.2,-.2,.7,.1,.4]])
        pose,blend=crouched_posture(q,self.names,2.)
        self.assertEqual(blend,1.)
        self.assertAlmostEqual(pose[0,1].item(),.5)
        torch.testing.assert_close(pose[:,:3].sum(1),q[:,:3].sum(1))
        torch.testing.assert_close(pose[:,3:],q[:,3:])
        self.assertAlmostEqual(q[0,1].item(),-.03)

    def test_starts_at_source_and_transitions_without_initial_velocity_jump(self):
        q=torch.zeros(1,7)
        start,_=crouched_posture(q,self.names,0.)
        near,_=crouched_posture(q,self.names,.001)
        torch.testing.assert_close(start,q)
        self.assertLess(float((near-start).norm()/.001),1e-5)

    def test_analytic_crouch_derivatives_include_blend_and_source_motion(self):
        q=torch.tensor([[.1,.1,.2,-.2,.7,.1,.4]],dtype=torch.float64)
        v=torch.full_like(q,.1);a=torch.full_like(q,-.03);t=.6;dt=1e-4
        pose,velocity,acceleration,_=crouched_posture_kinematics(q,v,a,self.names,t)
        before=crouched_posture(q-dt*v+.5*dt**2*a,self.names,t-dt)[0]
        after=crouched_posture(q+dt*v+.5*dt**2*a,self.names,t+dt)[0]
        torch.testing.assert_close(velocity,(after-before)/(2*dt),atol=1e-7,rtol=1e-6)
        torch.testing.assert_close(acceleration,(after-2*pose+before)/dt**2,atol=1e-7,rtol=1e-6)


if __name__=='__main__':unittest.main()
