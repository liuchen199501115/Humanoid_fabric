"""A pre-QP failure must retain landing evidence without claiming success."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from test_g1 import save_reference_failure

class ReferenceFailureReportTest(unittest.TestCase):
    def test_verified_landing_survives_failed_controller(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'trial.npz'
            args=SimpleNamespace(output_path=output,source_frame_count=30,
                                 source_start_frame=0,track_full_motion=True)
            plan=dict(continuous=True,full_motion=True,landing_goal_policy='source_pose',
                      events=[dict(support=1,takeoff_frame=0,end_frame=1,landing_window_end_frame=20)],
                      landing_positions_world=[[0.,0.,0.]],
                      collision_points=[[[0.,0.,0.]],[[0.,0.,0.]]],collision_radii=[[0.],[0.]])
            output.with_suffix('.steps_plan.json').write_text(json.dumps(plan))
            q=np.zeros((1,36));q[:,6]=1.;q[:,2]=.7
            history=[]
            for i in range(31):
                feet=np.zeros((1,2,7));feet[:,:,6]=1.;feet[0,0,2]=.03 if i==0 else 0.
                forces=np.zeros((1,2,3));forces[0,:,2]=[0.,150.] if i==0 else [100.,100.]
                history.append(dict(time=i*.002,frame=i*.2,q=q.copy(),reference_q=q.copy(),
                                    feet=feet,ground_foot_forces=forces,measured_mode=np.array([2 if i==0 else 3]),
                                    source_feet_reference=np.zeros((1,2,3))))
            with contextlib.redirect_stdout(io.StringIO()):
                save_reference_failure(args,ValueError('right ankle cannot stop'),.062,
                                       torch.tensor(q),torch.zeros(1,35),torch.ones(1,2),history)
            steps=json.loads(output.with_suffix('.steps.json').read_text())
            tracking=json.loads(output.with_suffix('.tracking.json').read_text())
            self.assertEqual(steps['measured_steps'],1)
            self.assertTrue(steps['pkl_fidelity_passed'])
            self.assertFalse(steps['controller_valid'])
            self.assertFalse(steps['steps_tracking_passed'])
            self.assertFalse(tracking['full_motion_tracking_passed'])
            self.assertEqual(tracking['measured_landings'],1)
            self.assertEqual(tracking['failed_time_s'],.062)
            self.assertEqual(tracking['stop_reason'],'right ankle cannot stop')
            self.assertLess(tracking['last_source_frame'],29)

    def test_empty_history_still_preserves_failed_state(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'trial.npz';args=SimpleNamespace(output_path=output)
            with contextlib.redirect_stdout(io.StringIO()):
                save_reference_failure(args,ValueError('startup'),0.,torch.zeros(1,36),
                                       torch.zeros(1,35),torch.zeros(1,2),[])
            with np.load(output.with_suffix('.reference_failure.npz')) as recording:
                self.assertEqual(str(recording['reason']),'startup')
                self.assertFalse(bool(recording['motion_tracking_valid']))
            self.assertFalse(output.with_suffix('.steps.json').exists())

if __name__=='__main__':unittest.main()
