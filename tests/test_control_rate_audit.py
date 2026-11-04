import unittest
import numpy as np
from analyze_control_rate import summarize_control_rate


class ControlRateTests(unittest.TestCase):
    def data(self):
        t=np.arange(100)*.002
        return dict(history_time=t,servo_time=t,servo_physics_time=t+1e-10*np.arange(100),
                    servo_qp_update_index=np.arange(100),history_qp_wall_seconds=np.full(100,.01))

    def test_500_hz_cadence_does_not_imply_realtime_solver(self):
        result=summarize_control_rate(self.data())
        self.assertTrue(result['simulation_rate_passed'])
        self.assertFalse(result['all_qp_solves_within_period'])

    def test_repeated_qp_or_extra_physics_steps_fail(self):
        for key,value in [('servo_qp_update_index',np.arange(100)//10),
                          ('servo_physics_time',np.arange(100)*.02),
                          ('history_time',np.arange(100)*.02)]:
            data=self.data();data[key]=value
            self.assertFalse(summarize_control_rate(data)['simulation_rate_passed'])

    def test_missing_torque_history_is_not_a_pass(self):
        self.assertFalse(summarize_control_rate({'history_time':np.arange(10)})['simulation_rate_passed'])
