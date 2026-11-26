import unittest
from preview_warm_stall import contact_restoration_stalled

def history(gaps):
    return [dict(iteration=i,elastic_slacks={'future_contact_position':gap},nonlinear_support_gap=gap,
                 nonlinear_collision_barrier=.001,nonlinear_base_force_residual=.1,
                 nonlinear_base_moment_residual=.1,nonlinear_joint_residual=.1,
                 nonlinear_integration_position=1e-6,nonlinear_integration_orientation=1e-6,
                 nonlinear_integration_joint=1e-6) for i,gap in enumerate(gaps)]
class WarmStallTests(unittest.TestCase):
    def test_plateau_requests_cold_fallback(self):
        self.assertTrue(contact_restoration_stalled(history([8.4e-5,8.5e-5,8.3e-5,8.6e-5]),[]))
    def test_improving_or_feasible_contact_is_not_stalled(self):
        for gaps in ([2e-4,1e-4,8e-5,6e-5],[6e-5,5.5e-5,5.1e-5,4.9e-5]):
            self.assertFalse(contact_restoration_stalled(history(gaps),[]))
    def test_recent_patch_change_or_other_residual_preserves_iterations(self):
        rows=history([8e-5]*4)
        self.assertFalse(contact_restoration_stalled(rows,[dict(iteration=2)]))
        rows[-1]['nonlinear_collision_barrier']=.1
        self.assertFalse(contact_restoration_stalled(rows,[]))
if __name__=='__main__':unittest.main()
