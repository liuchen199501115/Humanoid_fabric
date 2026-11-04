import tempfile
import unittest
from pathlib import Path
import numpy as np
from inspect_mocap_reference import urdf_fk, contact_segments

class MocapAuditTests(unittest.TestCase):
    def test_fk_rotates_joint_then_fixed_com_offset(self):
        xml='''<robot name="test"><link name="root"><inertial><mass value="1"/></inertial></link>
        <link name="arm"><inertial><mass value="1"/><origin xyz="1 0 0"/></inertial></link>
        <link name="tip"/><joint name="hinge" type="revolute"><parent link="root"/><child link="arm"/>
        <origin xyz="1 0 0"/><axis xyz="0 0 1"/></joint>
        <joint name="fixed" type="fixed"><parent link="arm"/><child link="tip"/><origin xyz="1 0 0"/></joint></robot>'''
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'robot.urdf';path.write_text(xml)
            q=np.array([[2,0,0,0,0,0,1,np.pi/2]])
            poses,com,mass=urdf_fk(q,['hinge'],path)
            np.testing.assert_allclose(poses['tip'][0],[[3,1,0]],atol=1e-12)
            np.testing.assert_allclose(com,[[2.5,.5,0]],atol=1e-12)
            self.assertEqual(mass,2)
            with self.assertRaises(ValueError):urdf_fk(q,['other'],path)

    def test_slip_segments_do_not_bridge_swing(self):
        mask=np.array([[1,0],[1,0],[0,0],[1,0]],bool)
        points=np.zeros((4,2,3));points[:,0,0]=[0,.01,3,4]
        result=contact_segments(mask,points,10)
        self.assertEqual(len(result),2)
        self.assertAlmostEqual(result[0]['displacement_m'],.01)
        self.assertEqual(result[1]['displacement_m'],0)
        self.assertEqual(result[0]['stop_exclusive'],2)
