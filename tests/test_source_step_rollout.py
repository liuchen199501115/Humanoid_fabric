import unittest
from types import SimpleNamespace
import torch
from fabrics_sim.reference.source_step_rollout import SourceStepRollout


class SourceStepRolloutTests(unittest.TestCase):
    def rollout(self):
        r = object.__new__(SourceStepRollout)
        r.support = 0
        r.touchdown = r.ever_clear = False
        r.touchdown_count = 0
        r.duration, r.dt = 1.3, .002
        r.rows = [{'frame': 114}, {'frame': 129}]
        r.references = ['start', 'end']
        return r

    def test_strict_landing_rejects_hidden_target_adaptations(self):
        for options in ({'landing_fraction': .5}, {'landing_search_depth': .004}):
            with self.assertRaisesRegex(ValueError, 'Strict source landing'):
                SourceStepRollout(None, None, [], None, 114, 129, 0, .002, None, None,
                                  candidate_options=options, strict_landing=True)
        with self.assertRaisesRegex(ValueError, 'Strict source landing'):
            SourceStepRollout(None, None, [], None, 114, 129, 0, .002, None, None,
                              maximum_step_distance=.3, strict_landing=True)

    def test_unknown_candidate_option_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown source-step'):
            SourceStepRollout(None, None, [], None, 114, 129, 0, .002, None, None,
                              candidate_options={'disable_joint_limits': True})

    def test_live_constructor_disables_autograd_and_restores_caller_mode(self):
        class Probe:
            def get_link_index(self, name):
                if torch.is_grad_enabled():
                    raise AssertionError('Planner inherited simulation autograd')
                raise RuntimeError('checked no-grad boundary')
        fabric = SimpleNamespace(dynamics_tasks=SimpleNamespace(model=SimpleNamespace(kinematics=Probe())))
        with torch.enable_grad():
            with self.assertRaisesRegex(RuntimeError, 'checked no-grad boundary'):
                SourceStepRollout(fabric, None, [], None, 114, 129, 0, .002, None, None)
            self.assertTrue(torch.is_grad_enabled())

    def test_requires_clearance_then_three_measured_flat_loaded_samples(self):
        r = self.rollout()
        for _ in range(5):
            self.assertEqual(r.contact_mode([300, 20], [0, 0], [0, 0]), 1)
        self.assertEqual(r.contact_mode([300, 0], [0, .04], [0, .04]), 1)
        for _ in range(5):
            self.assertEqual(r.contact_mode([300, 0], [0, 0], [0, 0]), 1)
        for confirmed in (False, False, True):
            self.assertEqual(r.contact_mode([300, 20], [0, 0], [0, .001]), 3)
            self.assertEqual(r.touchdown, confirmed)
        with self.assertRaises(RuntimeError):
            r.contact_mode([300, 0], [0, .01], [0, .01])

    def test_transient_contact_activates_constraint_without_certifying_landing(self):
        r = self.rollout()
        r.contact_mode([300,0], [0,.04], [0,.04])
        self.assertEqual(r.contact_mode([300,500], [0,0], [0,.001]),3)
        self.assertFalse(r.touchdown)
        self.assertEqual(r.contact_mode([300,0], [0,.003], [0,.004]),1)
        self.assertFalse(r.touchdown)
        self.assertEqual(r.touchdown_count,0)
        for _ in range(2):
            self.assertEqual(r.contact_mode([300,20], [0,0], [0,.001]),3)
            self.assertFalse(r.touchdown)
        r.contact_mode([300,20], [0,0], [0,.001])
        self.assertTrue(r.touchdown)

    def test_stop_requires_confirmed_landing_and_measured_support(self):
        r=self.rollout();r.swing_duration=1.;r.stop_duration=.3
        self.assertTrue(r.has_measured_support([True,False]))
        self.assertFalse(r.has_measured_support([False,True]))
        with self.assertRaisesRegex(RuntimeError,'confirmed measured touchdown'):
            r.sample(1.1)
        r.touchdown=True
        self.assertTrue(r.has_measured_support([False,True]))
        self.assertFalse(r.has_measured_support([False,False]))
        self.assertEqual(r.sample(1.1),'end')

    def test_heel_contact_does_not_certify_flat_foot(self):
        r = self.rollout()
        r.contact_mode([300, 0], [0, .04], [0, .04])
        for _ in range(5):
            self.assertEqual(r.contact_mode([300, 20], [0, 0], [0, .02]), 1)
        self.assertFalse(r.touchdown)

    def test_invalid_observation_and_unplanned_continuation_rejected(self):
        r = self.rollout()
        with self.assertRaises(ValueError):
            r.contact_mode([300, float('nan')], [0, 0], [0, 0])
        with self.assertRaises(RuntimeError):
            r.sample(1.31)
        with self.assertRaises(ValueError):
            r.sample(-.1)
