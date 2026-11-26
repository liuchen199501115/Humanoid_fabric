import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from g1_joint_braking import named_braking_coordinates


class NamedBrakingTests(unittest.TestCase):
    def test_physical_joint_order_and_base_offset(self):
        joints, accelerations = named_braking_coordinates(
            ['hip', 'ankle', 'waist', 'shoulder'], ['waist', 'ankle'])
        self.assertEqual(joints, [2, 1])
        self.assertEqual(accelerations, [8, 7])


    def test_invalid_selection_fails_before_control(self):
        for selected in [[], ['waist', 'waist'], ['missing']]:
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                named_braking_coordinates(['hip', 'waist'], selected)


if __name__ == '__main__':
    unittest.main()
