"""Execution-mode guards and continuity of the retained support unloading."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
import test_g1


class DirectFlowTest(unittest.TestCase):
    def options(self, *extra):
        with patch.object(sys, 'argv', ['test_g1.py', '--tracking-profile', 'amp-full', *extra]):
            return test_g1.parse_args()

    def test_current_flow_allows_short_runs_and_snapshots(self):
        args = self.options('--duration', '1', '--diagnose-qp-time', '.2')
        test_g1.validate_direct_wbc_flow(args)

    def test_asynchronous_servo_and_archived_modes_fail_before_isaac(self):
        for extra in [('--control-frequency', '250'), ('--physics-frequency', '1000'),
                      ('--fixed-joint-test',), ('--reference-manager',),
                      ('--waist-feedback-start', '0')]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                test_g1.validate_direct_wbc_flow(self.options(*extra))

    def test_unload_ramp_retains_its_start_across_active_ticks(self):
        # Exercise the actual loop block without starting Isaac. A conditional
        # reset must not become unconditional when inactive modes are archived.
        module = ast.parse(Path(test_g1.__file__).read_text())
        main = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        block = next(n for n in ast.walk(main) if isinstance(n, ast.If)
                     and isinstance(n.test, ast.Name) and n.test.id == 'dynamic_unloading')
        code = compile(ast.Module(body=[block], type_ignores=[]), test_g1.__file__, 'exec')
        state = dict(torch=torch, args=SimpleNamespace(motion_start_in_motion=True,
                     late_unload_start=3.7, late_unload_ramp=.02),
                     q=torch.zeros(1, 36), device='cpu', dynamic_unloading=True,
                     unload_intended_mode=1, direct_unload_event=None)
        weights = []
        for tick in range(3):
            state.update(control_time=.002*tick, elapsed_motion_time=.002*tick)
            exec(code, state)
            weights.append(float(state['wrench_weights'].max()))
            self.assertEqual(state['direct_unload_event'], 1)
            self.assertEqual(state['direct_unload_started'], 0.)
        self.assertEqual(weights[0], 0.)
        self.assertGreater(weights[1], 0.)
        self.assertGreater(weights[2], weights[1])
        state['dynamic_unloading'] = False
        exec(code, state)
        self.assertIsNone(state['direct_unload_event'])


if __name__ == '__main__':
    unittest.main()
