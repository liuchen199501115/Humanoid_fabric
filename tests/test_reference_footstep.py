import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from fabrics_sim.reference.footstep import grounded_step, source_grounded_step, source_com_landing_target, source_toe_pivot_pose, source_contact_step


class FootstepTests(unittest.TestCase):
    def test_source_contact_matches_tilted_swing_endpoint(self):
        rotation=Rotation.from_euler('zyx',[.7,-.25,.04])
        source=np.r_[12.,-8.,.06,rotation.as_quat()];before=source.copy()
        goal=source_contact_step(source)
        np.testing.assert_array_equal(source,before)
        np.testing.assert_array_equal(goal[:2],source[:2])
        np.testing.assert_array_equal(goal[3:],source[3:])
        corners=np.array([[-.05,-.025,-.035],[-.05,.025,-.035],[.12,-.025,-.035],[.12,.025,-.035]])
        world=rotation.apply(corners)+goal[:3]
        self.assertAlmostEqual(world[:,2].min(),0.,places=12)
        self.assertGreater(world[:,2].max(),.02)  # Heel landing, not a flattened foot.
        with self.assertRaisesRegex(ValueError,'4 cm'):
            source_contact_step(source+np.array([0.,0.,.2,0.,0.,0.,0.]))

    def test_rollover_pose_preserves_material_toe_and_stance_heading(self):
        stance=np.r_[2.,-3.,.035,Rotation.from_euler('z',.7).as_quat()]
        source=np.r_[2.,-3.,.1,(Rotation.from_euler('z',.9)*Rotation.from_euler('y',.5)).as_quat()]
        goal=source_toe_pivot_pose(stance,source);toe=np.array([.12,0.,-.035])
        np.testing.assert_allclose(goal[:3]+Rotation.from_quat(goal[3:]).apply(toe),
            stance[:3]+Rotation.from_quat(stance[3:]).apply(toe),atol=1e-12)
        R=Rotation.from_quat(goal[3:]).as_matrix()
        self.assertAlmostEqual(np.arctan2(R[1,0],R[0,0]),.7)
        self.assertAlmostEqual(np.arcsin(-R[2,0]),.5)
        self.assertGreater(goal[2],stance[2])

    def test_com_endpoint_retains_forward_offset_in_heading_frame(self):
        feet=np.array([[0.,.1,.035,0.,0.,0.,1.],[.6,-.1,.035,0.,0.,0.,1.]])
        # Source COM is 4 cm ahead of its feet midpoint, with lateral sway.
        com=np.array([.34,.06,.65]);stance=feet[0].copy();stance[0]=-.1
        result=source_com_landing_target(com,feet,stance,feet[1])
        np.testing.assert_allclose(result,[.29,0.],atol=1e-12)
        rotation=Rotation.from_euler('z',.8);shift=np.array([3.,-2.,0.])
        rotated=feet.copy();rotated[:,:3]=rotation.apply(feet[:,:3])+shift
        rotated[:,3:]=rotation.as_quat()
        actual=rotated[0].copy();actual[:3]=rotation.apply(stance[:3])+shift
        value=source_com_landing_target(rotation.apply(com)+shift,rotated,actual,rotated[1])
        np.testing.assert_allclose(value,(rotation.apply(np.r_[result,0.])+shift)[:2],atol=1e-12)

    def test_source_step_preserves_world_path_and_yaw(self):
        rotation=Rotation.from_euler('zyx',[.7,.2,-.1])
        source=np.r_[12.,-8.,.08,rotation.as_quat()]
        before=source.copy()
        goal=source_grounded_step(source)
        np.testing.assert_array_equal(source,before)
        np.testing.assert_allclose(goal[:3],[12.,-8.,.035])
        matrix=rotation.as_matrix()
        expected=Rotation.from_euler('z',np.arctan2(matrix[1,0],matrix[0,0])).as_matrix()
        np.testing.assert_allclose(Rotation.from_quat(goal[3:]).as_matrix(),expected,atol=1e-8)

    def test_relative_step_caps_forward_and_keeps_side(self):
        support=np.array([3.,4.,.04,0.,0.,0.,1.])
        swing=support.copy();swing[:3]+=[.5,-.2,.02]
        actual=support.copy();actual[:3]+=[10.,20.,0.]
        goal=grounded_step(support,swing,actual)
        np.testing.assert_allclose(goal[:3], [13.16,23.82,.035])
        np.testing.assert_allclose(goal[3:], [0.,0.,0.,1.])

    def test_world_heading_change_rotates_step(self):
        support=np.array([0.,0.,.035,0.,0.,0.,1.])
        swing=support.copy();swing[:2]=[.15,.18]
        actual=support.copy();actual[3:]=Rotation.from_euler('z',np.pi/2).as_quat()
        goal=grounded_step(support,swing,actual)
        np.testing.assert_allclose(goal[:3],[-.18,.15,.035],atol=1e-8)
        np.testing.assert_allclose(goal[3:],actual[3:],atol=1e-8)

if __name__=='__main__':unittest.main()
