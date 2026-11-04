import unittest

import torch

from fabrics_sim.reference.interpolation import MotionInterpolator


class MotionInterpolationTests(unittest.TestCase):
    def test_cubic_has_continuous_acceleration_at_internal_knots(self):
        q=self.trajectory()
        sample=MotionInterpolator(q,method='cubic')
        for frame in (1.,2.,3.):
            before=sample.kinematics(frame-1e-7,2.,.3)
            after=sample.kinematics(frame+1e-7,2.,.3)
            torch.testing.assert_close(before[1],after[1],atol=1e-5,rtol=1e-5)
            torch.testing.assert_close(before[2],after[2],atol=1e-5,rtol=1e-5)
            torch.testing.assert_close(sample(frame),q[int(frame):int(frame)+1])
        with self.assertRaises(ValueError):
            MotionInterpolator(q,method='unknown')

    def test_foot_pose_has_consistent_six_dimensional_twist(self):
        q=self.trajectory()[:,:7]
        interpolate=MotionInterpolator(q,pose_dimension=7)
        pose,v,a=interpolate.kinematics(2.3,15.,-2.)
        self.assertEqual(pose.shape,(1,7))
        self.assertEqual(v.shape,(1,6))
        self.assertEqual(a.shape,(1,6))
        torch.testing.assert_close(v[0,:3],torch.tensor([1.5,0.,0.],dtype=q.dtype))
        torch.testing.assert_close(a[0,:3],torch.tensor([-.2,0.,0.],dtype=q.dtype))
        stopped=MotionInterpolator(q[:1],pose_dimension=7).kinematics(0.,0.)
        self.assertEqual(stopped[1].shape,(1,6))

    def trajectory(self):
        q = torch.zeros(5,36,dtype=torch.float64)
        q[:,6] = 1
        q[:,7] = torch.tensor([0.,.2,.8,.5,.6])
        q[:,0] = torch.arange(5,dtype=q.dtype)*.1
        return q

    def test_all_frames_preserved_and_joint_positions_do_not_overshoot(self):
        q = self.trajectory(); interpolate = MotionInterpolator(q)
        for i in range(5):
            torch.testing.assert_close(interpolate(i),q[i:i+1],rtol=0,atol=0)
        for i in range(4):
            for offset in (.1,.25,.5,.75,.9):
                value = interpolate(i+offset)[0,7:]
                self.assertTrue((value>=torch.minimum(q[i,7:],q[i+1,7:])-1e-12).all())
                self.assertTrue((value<=torch.maximum(q[i,7:],q[i+1,7:])+1e-12).all())

    def test_velocity_continuous_at_interior_knots(self):
        interpolate=MotionInterpolator(self.trajectory());eps=1e-5
        for i in (1,2,3):
            before=(interpolate(i)-interpolate(i-eps))/eps
            after=(interpolate(i+eps)-interpolate(i))/eps
            torch.testing.assert_close(before,after,atol=5e-5,rtol=0)

    def test_quaternion_sign_changes_do_not_create_a_rotation(self):
        q=self.trajectory();q[1::2,6]=-1
        interpolate=MotionInterpolator(q)
        for frame in (.5,1.5,2.5,3.5):
            quat=interpolate(frame)[0,3:7]
            self.assertAlmostEqual(float(quat.norm()),1.)
            self.assertAlmostEqual(float(quat[3].abs()),1.)
        torch.testing.assert_close(q[1:2],interpolate(1),rtol=0,atol=0)

    def test_single_frame_and_invalid_input(self):
        q=self.trajectory()[:1];interpolate=MotionInterpolator(q)
        torch.testing.assert_close(interpolate(3),q)
        with self.assertRaises(ValueError):interpolate(float('nan'))
        with self.assertRaises(ValueError):MotionInterpolator(torch.zeros(2,36))

    def test_analytic_derivatives_include_clock_acceleration_and_rotation(self):
        from scipy.spatial.transform import Rotation
        q=self.trajectory()
        for i in range(len(q)):
            q[i,3:7]=torch.tensor(Rotation.from_rotvec([.03*i,.02*i*i,-.04*i]).as_quat())
        interpolate=MotionInterpolator(q)
        frame,rate,acceleration,eps=1.4,1.7,-.6,1e-4
        pose,v,a=interpolate.kinematics(frame,rate,acceleration)
        before=interpolate(frame-rate*eps+.5*acceleration*eps**2)
        after=interpolate(frame+rate*eps+.5*acceleration*eps**2)
        columns=[0,1,2]+list(range(7,36));velocity_columns=[0,1,2]+list(range(6,35))
        torch.testing.assert_close(v[:,velocity_columns],(after-before)[:,columns]/(2*eps),atol=1e-7,rtol=1e-6)
        torch.testing.assert_close(a[:,velocity_columns],(after-2*pose+before)[:,columns]/eps**2,atol=1e-7,rtol=1e-6)
        omega=(Rotation.from_quat(after[0,3:7])*Rotation.from_quat(before[0,3:7]).inv()).as_rotvec()/(2*eps)
        torch.testing.assert_close(v[0,3:6],torch.tensor(omega),atol=1e-7,rtol=1e-6)
        vm=interpolate.kinematics(frame-rate*eps+.5*acceleration*eps**2,rate-acceleration*eps,acceleration)[1]
        vp=interpolate.kinematics(frame+rate*eps+.5*acceleration*eps**2,rate+acceleration*eps,acceleration)[1]
        torch.testing.assert_close(a,(vp-vm)/(2*eps),atol=1e-7,rtol=1e-6)

    def test_float32_translation_has_no_second_difference_noise(self):
        q=self.trajectory().float();q[:,1]=4.5+torch.arange(5)*.03125
        sample=MotionInterpolator(q)
        _,v,a=sample.kinematics(1.37,30.)
        self.assertEqual(float(v[0,1]),.9375)
        self.assertEqual(float(a[0,1]),0.)
