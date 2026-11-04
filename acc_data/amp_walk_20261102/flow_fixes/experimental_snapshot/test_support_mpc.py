import unittest
import numpy as np
from fabrics_sim.reference.support_mpc import SupportMPC

class SupportMPCTests(unittest.TestCase):
    def test_static_support_is_equilibrium(self):
        m=SupportMPC(.7)
        polygon=np.array([[-.1,-.04],[-.1,.04],[.1,-.04],[.1,.04]])
        u=m.solve([0,0],[0,0],np.zeros((m.horizon,2)),[polygon]*m.horizon)
        np.testing.assert_allclose(u,0,atol=1e-7)
        self.assertTrue(m.last_report['valid'])

    def test_cop_remains_inside_support_when_reference_is_outside(self):
        m=SupportMPC(.7)
        polygon=np.array([[-.1,-.04],[-.1,.04],[.1,-.04],[.1,.04]])
        u=m.solve([0,0],[0,0],np.tile([.5,.2],(m.horizon,1)),[polygon]*m.horizon)
        cop=.5*m.dt**2*u-u/m.omega2
        self.assertLessEqual(abs(cop[0]),.1+1e-4)
        self.assertLessEqual(abs(cop[1]),.04+1e-4)

if __name__=='__main__':unittest.main()
