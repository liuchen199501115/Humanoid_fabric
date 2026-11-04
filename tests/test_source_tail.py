import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
import numpy as np
import torch
import warp as wp
from fabrics_sim.reference.source_tail import DoubleSupportSourceRollout
from fabrics_sim.reference.whole_body import WholeBodyReference
from fabrics_sim.reference.source_sequence import next_source_event


class SourceEventTests(unittest.TestCase):
    def test_repeated_side_and_direct_switch_keep_source_boundaries(self):
        modes=[3,1,1,3,1,1,2,2,3,3]
        first=next_source_event(modes,0)
        second=next_source_event(modes,first['end_frame'])
        third=next_source_event(modes,second['end_frame'])
        tail=next_source_event(modes,third['end_frame'])
        self.assertEqual([e['support'] for e in (first,second,third)],[0,0,1])
        self.assertEqual([(e['start_frame'],e['end_frame']) for e in (first,second,third,tail)],
                         [(0,3),(3,6),(6,8),(8,9)])
        self.assertTrue(second['inserted_double_support'])
        self.assertEqual(tail['kind'],'tail')

    def test_unfinished_or_missing_contact_events_are_rejected(self):
        for modes,frame in (([3,1,1,3],2),([3,0,3],0),([3,1,1],0),([3,1,3],.5)):
            with self.assertRaises(ValueError):next_source_event(modes,frame)


class SourceTailTests(unittest.TestCase):
    @torch.no_grad()
    def test_tail_preserves_feet_and_reaches_last_frame_with_zero_rates(self):
        from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
        from test_g1_two_tasks import G1_JOINT_NAMES
        from g1_armature import unitree_armatures
        wp.config.kernel_cache_dir='/tmp/g1_warp_cache';wp.init()
        f=G1PoseFabric(1,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware')
        data=json.loads((Path(__file__).parent/'fixtures/g1_recovery_support_switch.json').read_text())
        q=torch.tensor(data['q']);zero=torch.zeros(1,35);ref=WholeBodyReference(q,zero,zero)
        source=q.repeat(3,1);source[:,19]+=torch.tensor([0.,.01,.02]);original=source.clone()
        limits={n.attrib['name']:n.find('limit') for n in ET.parse(f.urdf_path).getroot().findall('joint')}
        bounds=dict(lower=q.new_tensor([float(limits[n].attrib['lower']) for n in G1_JOINT_NAMES]),
                    upper=q.new_tensor([float(limits[n].attrib['upper']) for n in G1_JOINT_NAMES]),dt=.002,bandwidth=8.)
        with tempfile.TemporaryDirectory() as tmp:
            plan=DoubleSupportSourceRollout(f,ref,G1_JOINT_NAMES,source,0,2,1,.002,
                q.new_tensor(unitree_armatures(G1_JOINT_NAMES)),Path(tmp)/'tail.json',bounds,5*f.accel_limits)
            self.assertTrue(json.loads((Path(tmp)/'tail.json').read_text())['sampled_necessary_constraints_passed'])
        self.assertEqual((plan.frame(0.),plan.frame(1.)),(0.,2.))
        self.assertTrue(np.all(np.diff([r['frame'] for r in plan.rows])>=0))
        torch.testing.assert_close(source,original)
        torch.testing.assert_close(plan.references[0].q,q,atol=3e-6,rtol=0.)
        end=plan.references[-1]
        torch.testing.assert_close(end.q[:,19:],source[-1:,19:])
        torch.testing.assert_close(end.v,zero,atol=1e-6,rtol=0.)
        torch.testing.assert_close(end.a,zero,atol=1e-6,rtol=0.)
        feet=[f.dynamics_tasks.model.kinematics.get_link_index(s+'_ankle_roll_link') for s in ('left','right')]
        begin=f.dynamics_tasks.model.snapshot(q)['transforms'][:,feet]
        for time in (.2,.6,1.):
            actual=f.dynamics_tasks.model.snapshot(plan.sample(time).q)['transforms'][:,feet]
            torch.testing.assert_close(actual,begin,atol=3e-6,rtol=0.)
        self.assertEqual(plan.contact_mode([150.,150.],[0.,0.],[0.,0.]),3)
        with self.assertRaises(RuntimeError):plan.contact_mode([0.,0.],[0.,0.],[0.,0.])
        with self.assertRaises(RuntimeError):plan.contact_mode([300.,0.],[0.,.03],[0.,.03])
