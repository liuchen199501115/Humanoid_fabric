import unittest
import numpy as np
from audit_mocap_dynamics import allocate_wrenches

class WrenchAllocationTests(unittest.TestCase):
    def arrays(self):
        jac=np.zeros((12,35));jac[:6,:6]=np.eye(6);jac[6:,:6]=np.eye(6)
        gen=np.zeros(35);gen[2]=100
        return jac,gen,np.zeros((0,12)),np.zeros(0),np.ones(29)
    def test_support_satisfies_floating_base_balance(self):
        r=allocate_wrenches(*self.arrays(),[True,False])
        self.assertTrue(r['feasible']);self.assertEqual(r['peak_torque_ratio'],0)
    def test_inactive_foot_cannot_supply_force(self):
        jac,gen,g,b,eff=self.arrays();jac[:6,:6]=0
        self.assertFalse(allocate_wrenches(jac,gen,g,b,eff,[True,False])['feasible'])
        self.assertTrue(allocate_wrenches(jac,gen,g,b,eff,[False,True])['feasible'])
    def test_torque_overload_is_reported_and_flight_rejected(self):
        jac,gen,g,b,eff=self.arrays();gen[6]=2
        self.assertEqual(allocate_wrenches(jac,gen,g,b,eff,[True,True])['peak_torque_ratio'],2)
        self.assertFalse(allocate_wrenches(jac,gen,g,b,eff,[False,False])['feasible'])
