import unittest
from fabrics_sim.reference.source_step_optimization import relative_peak_objective


class RelativePeakTests(unittest.TestCase):
    def test_one_family_cannot_hide_another_regression(self):
        baseline=dict(wrench=.2,braking=.02,torque=0.)
        improved=dict(wrench=.18,braking=.018,torque=0.)
        regressed=dict(wrench=.1,braking=.022,torque=0.)
        self.assertLess(relative_peak_objective(improved,baseline),1.)
        self.assertGreater(relative_peak_objective(regressed,baseline),1.)
        self.assertEqual(relative_peak_objective(dict.fromkeys(baseline,0.),baseline),0.)

    def test_previously_satisfied_constraints_remain_protected(self):
        baseline=dict(wrench=.2,torque=0.)
        self.assertGreater(relative_peak_objective(dict(wrench=0.,torque=.01),baseline),1.)

    def test_invalid_peaks_rejected(self):
        for value in (-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):relative_peak_objective(dict(a=value),dict(a=.1))
        with self.assertRaises(ValueError):relative_peak_objective(dict(a=.1),dict(b=.1))
