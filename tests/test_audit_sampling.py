import unittest
import numpy as np
from fabrics_sim.reference.audit_sampling import refined_sample_times


class AuditSamplingTests(unittest.TestCase):
    def report(self):
        return dict(options=dict(duration=1.5), optimization_sample_times=[.731],
                    best=dict(sample_count=601, worst_sample=dict(time=.3),
                              worst_constraint_times=dict(joint_bound_score=1.2875),
                              collision=dict(worst_time=1.4, violation_times=[1.4025])))

    def test_preserves_endpoints_failures_neighbours_and_previous_grid(self):
        times=refined_sample_times(1.5,41,self.report())
        self.assertEqual(times[0],0.)
        self.assertEqual(times[-1],1.5)
        self.assertTrue(np.all(np.diff(times)>0))
        for t in (.731,.3,1.2875,1.4,1.4025,1.3975,1.405):
            self.assertTrue(np.any(np.isclose(times,t,atol=1e-12,rtol=0)))

    def test_rejects_stale_or_invalid_report_times(self):
        report=self.report();report['options']['duration']=2.
        with self.assertRaises(ValueError):refined_sample_times(1.5,41,report)
        for invalid in (float('nan'),-1.,2.):
            report=self.report();report['best']['worst_sample']['time']=invalid
            with self.assertRaises(ValueError):refined_sample_times(1.5,41,report)

    def test_legacy_report_and_uniform_grid(self):
        report=dict(options=dict(duration=1.5),best=dict(collision=dict(worst_time=None)))
        np.testing.assert_array_equal(refined_sample_times(1.5,4,report),[0.,.5,1.,1.5])
        np.testing.assert_array_equal(refined_sample_times(1.5,4),[0.,.5,1.,1.5])
