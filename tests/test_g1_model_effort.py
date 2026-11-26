import unittest
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
from g1_model_effort import model_effort_limits
URDF=Path(__file__).resolve().parents[1]/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'

class ModelEffortTests(unittest.TestCase):
    def test_active_model_parallel_actuators_and_named_order(self):
        names=['right_wrist_pitch_joint','waist_roll_joint','left_ankle_pitch_joint','right_knee_joint','left_shoulder_roll_joint']
        np.testing.assert_array_equal(model_effort_limits(URDF,names,[[5,50,50,139,25]]),[5,50,50,139,25])
    def test_neither_simulator_nor_urdf_limit_is_exceeded(self):
        names=['left_ankle_roll_joint','right_ankle_pitch_joint']
        np.testing.assert_array_equal(model_effort_limits(URDF,names,[23,80]),[23,50])
    def test_missing_duplicate_and_invalid_limits_rejected(self):
        for names,limits in [(['missing'],[1]),(['waist_roll_joint']*2,[50,50]),(['waist_roll_joint'],[float('nan')]),(['waist_roll_joint'],[-1]),(['waist_roll_joint'],[1,2])]:
            with self.subTest(names=names,limits=limits),self.assertRaises(ValueError):model_effort_limits(URDF,names,limits)
if __name__=='__main__':unittest.main()
