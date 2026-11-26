"""Independent LP checks on the measured point-contact friction section."""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'acc_data/amp_walk_20261103/feedback_followup'))
from inscribed_patch_wrench import inscribed_wrench_cone


class InscribedPointContactTest(unittest.TestCase):
    def test_directional_capacity_is_inside_coulomb_circle_and_matches_rays(self):
        mu = .6
        for rays in (8, 16):
            G = inscribed_wrench_cone([[0., 0., 0.]], mu, rays)
            generators = mu*np.c_[np.cos(np.arange(rays)*2*np.pi/rays),
                                  np.sin(np.arange(rays)*2*np.pi/rays)]
            for angle in np.linspace(0, 2*np.pi, 65):
                direction = np.array([np.cos(angle), np.sin(angle)])
                cost = np.r_[-direction, np.zeros(4)]
                fit = linprog(cost, A_ub=G, b_ub=np.zeros(len(G)),
                    A_eq=np.eye(6)[2:3], b_eq=[1.], bounds=[(None, None)]*6,
                    method='highs')
                self.assertTrue(fit.success, fit.message)
                self.assertAlmostEqual(-fit.fun, np.max(generators@direction), places=8)
                self.assertLessEqual(np.linalg.norm(fit.x[:2]), mu+1e-8)
                np.testing.assert_allclose(fit.x[3:], 0., atol=1e-8)

    def test_point_contact_does_not_invent_a_support_moment(self):
        G = inscribed_wrench_cone([[0., 0., 0.]], .6, 16)
        for axis in range(3, 6):
            for sign in (-1., 1.):
                cost = np.zeros(6); cost[axis] = sign
                fit = linprog(cost, A_ub=G, b_ub=np.zeros(len(G)),
                    A_eq=np.eye(6)[2:3], b_eq=[1.], bounds=[(None, None)]*6,
                    method='highs')
                self.assertTrue(fit.success, fit.message)
                self.assertAlmostEqual(fit.fun, 0., places=9)
