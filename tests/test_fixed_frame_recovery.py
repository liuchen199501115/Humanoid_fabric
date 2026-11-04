import unittest
import torch
from fabrics_sim.reference.source_step_rollout import SourceStepRollout


class FixedFrameRecoveryTests(unittest.TestCase):
    def test_invalid_fixed_frame_and_cap_are_rejected(self):
        source = torch.zeros(2,36)
        for frame,start,end in ((-1,-1,-1),(2,2,2),(1,0,1),(1,1,0),(1.,1,1)):
            with self.assertRaisesRegex(ValueError,'hold one valid source frame'):
                SourceStepRollout(None,None,[],source,start,end,1,.002,None,None,fixed_source_frame=frame)
        for cap in (0.,-.1,float('nan'),float('inf')):
            with self.assertRaisesRegex(ValueError,'finite and positive'):
                SourceStepRollout(None,None,[],source,0,1,1,.002,None,None,maximum_step_distance=cap)

    def test_removed_reduced_model_recovery_stop_is_rejected(self):
        source = torch.zeros(2,36)
        with self.assertRaisesRegex(ValueError, 'excludes the reduced-model landing stop'):
            SourceStepRollout(None,None,[],source,1,1,1,.02,None,None,
                stop_duration=.4,fixed_source_frame=1,maximum_step_distance=.42)
