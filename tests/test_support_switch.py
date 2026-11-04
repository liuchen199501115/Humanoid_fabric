import unittest
from types import SimpleNamespace
from fabrics_sim.reference.support_switch import SupportSwitchReference


class SupportSwitchTests(unittest.TestCase):
    def gate(self,support=1):
        gate=object.__new__(SupportSwitchReference)
        gate.support=support;gate.mode=3;gate.dt=.002;gate.ready_samples=0;gate.confirmation_samples=25
        gate.transfer=SimpleNamespace(release_time=None,start_time=0.,duration=2.,ready_for_release=lambda *args:True)
        return gate

    def test_both_sides_need_continuous_load_and_balance_before_release(self):
        for support in (0,1):
            gate=self.gate(support)
            forces=[10.,10.];forces[support]=300.
            for _ in range(24):
                self.assertEqual(gate.update(3.,forces,[True,True],None,None,None),3)
            gate.transfer.ready_for_release=lambda *args:False
            self.assertEqual(gate.update(3.,forces,[True,True],None,None,None),3)
            self.assertEqual(gate.ready_samples,0)
            gate.transfer.ready_for_release=lambda *args:True
            for _ in range(24):gate.update(3.,forces,[True,True],None,None,None)
            self.assertEqual(gate.update(3.,forces,[True,True],None,None,None),support+1)

    def test_no_release_while_old_foot_carries_too_much(self):
        gate=self.gate()
        for _ in range(100):self.assertEqual(gate.update(3.,[80.,240.],[True,True],None,None,None),3)
        with self.assertRaisesRegex(RuntimeError,'timed out'):
            gate.update(7.1,[80.,240.],[True,True],None,None,None)

    def test_loss_of_new_support_after_release_is_rejected(self):
        gate=self.gate();gate.mode=2;gate.transfer.release_time=3.
        self.assertEqual(gate.update(4.,[0.,300.],[False,True],None,None,None),2)
        with self.assertRaises(RuntimeError):gate.update(4.,[300.,0.],[True,False],None,None,None)
        with self.assertRaises(RuntimeError):gate.update(4.,[0.,0.],[False,False],None,None,None)
        with self.assertRaises(ValueError):gate.update(4.,[0.,float('nan')],[False,True],None,None,None)
