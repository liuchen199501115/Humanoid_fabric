import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import warp as wp
from fabrics_sim.reference.optimized_source_rollout import OptimizedSourceRollout
from fabrics_sim.reference.whole_body import WholeBodyReference


class OptimizedSourceRolloutTests(unittest.TestCase):
    def test_live_boundary_rebuild_and_physical_limits(self):
        from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
        from test_g1_two_tasks import G1_JOINT_NAMES, load_motion_trajectory, unitree_g1_effort_limits
        from g1_armature import unitree_armatures
        import xml.etree.ElementTree as ET
        wp.config.kernel_cache_dir = '/tmp/g1_warp_cache'
        wp.init()
        torch.set_grad_enabled(False)
        f = G1PoseFabric(1, 'cpu', robot_name='g1_29dof_rev_1_0', task_profile='dynamics-aware')
        f.dynamics_tasks.enable_wrist_hip_avoidance(include_upper_body=True)
        f.dynamics_tasks.wrist_hip_barrier = True
        f.fabric_params['joint_limits']['active'] = False
        with np.load('acc_data/walk1_settled_unload_20261021.npz') as d:
            initial = WholeBodyReference(*(torch.tensor(d['history_'+key][-1]) for key in ('q','qd','qdd_predicted')))
            shift = float(d['motion_ground_shift'])
        source, _ = load_motion_trajectory(Path('retarget/lafan/walk1_g1.pkl'))
        source[:, 2] += shift
        limits = {n.attrib['name']: n.find('limit') for n in ET.parse(f.urdf_path).getroot().findall('joint')}
        effort = source.new_tensor([float(limits[n].attrib['effort']) for n in G1_JOINT_NAMES])*.8
        config = Path('acc_data/walk1_settled_handoff_margin_dense_20261023.json')
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp)/'report.json'
            # The expanded format must rebuild the same audited motion from
            # parameter values, even if legacy option defaults differ.
            expanded = json.loads(config.read_text())
            values = expanded['best']['parameters']
            roll = expanded['options']['pelvis_roll_offset']
            bump = expanded['options']['pelvis_roll_bump']
            expanded['best']['parameters'] = values[:-2]+[roll,bump]+values[-2:]
            expanded['pelvis_parameters_enabled'] = True
            expanded['options']['pelvis_roll_offset'] = .25
            expanded['options']['pelvis_roll_bump'] = .15
            expanded_path = Path(tmp)/'expanded.json'
            expanded_path.write_text(json.dumps(expanded))
            with self.assertRaisesRegex(RuntimeError, 'physical sample checks'):
                OptimizedSourceRollout(f,initial,G1_JOINT_NAMES,source,.002,
                    source.new_tensor(unitree_armatures(G1_JOINT_NAMES)),effort,expanded_path,report,project_nominal_torque=True)
            result = json.loads(report.read_text())
            self.assertFalse(result['physical_sample_checks_passed'])
            self.assertFalse(result['checks']['swing_ground'])
            self.assertLess(result['minimum_swing_sole_height_m'],-.01)
            self.assertTrue(result['checks']['torque'])
            self.assertTrue(result['checks']['boundary'])
            self.assertTrue(result['checks']['landing'])
            self.assertGreater(result['conservative_score'], 0.)
            self.assertFalse(result['full_pkl_tracking'])
            self.assertFalse(result['acceleration_and_braking_limits_enabled'])
            # Old candidate passed against URDF effort, but must be rejected
            # against the motor table actually used by the online WBC.
            with self.assertRaisesRegex(RuntimeError, 'physical sample checks'):
                OptimizedSourceRollout(f,initial,G1_JOINT_NAMES,source,.002,
                    source.new_tensor(unitree_armatures(G1_JOINT_NAMES)),
                    source.new_tensor(unitree_g1_effort_limits())*.8,config,report)
            rejected=json.loads(report.read_text())
            self.assertFalse(rejected['checks']['torque'])
            self.assertGreater(rejected['peak_torque_ratio'], 1.8)
            ankle=G1_JOINT_NAMES.index('left_ankle_pitch_joint')
            self.assertAlmostEqual(rejected['effort_limits'][ankle],20.)

            # A clearance-corrected reference can enter the explicit torque
            # projection probe without certifying exact nominal feasibility.
            corrected = Path('acc_data/walk1_arm_clearance_dense_20261027.json')
            rollout = OptimizedSourceRollout(f,initial,G1_JOINT_NAMES,source,.002,
                source.new_tensor(unitree_armatures(G1_JOINT_NAMES)),
                source.new_tensor(unitree_g1_effort_limits())*.8,corrected,report,
                project_nominal_torque=True)
            accepted = json.loads(report.read_text())
            self.assertTrue(accepted['eligible_for_bounded_probe'])
            self.assertFalse(accepted['physical_sample_checks_passed'])
            self.assertFalse(accepted['checks']['torque'])
            for key in ('swing_ground','wrench','collision','boundary','landing'):
                self.assertTrue(accepted['checks'][key])
            for key in ('q','v','a'):
                torch.testing.assert_close(getattr(rollout.sample(0.),key),getattr(initial,key),atol=2e-6,rtol=1e-5)
            for side, sign in (('left',1),('right',-1)):
                j=7+G1_JOINT_NAMES.index(side+'_shoulder_roll_joint')
                self.assertAlmostEqual(float(rollout.references[-1].q[0,j]-initial.q[0,j]),sign*.08,places=5)
            self.assertTrue(torch.equal(rollout.goal,rollout.source_goal))

            with self.assertRaisesRegex(ValueError, 'unscaled'):
                bad=json.loads(config.read_text());bad['options']['landing_fraction']=.8
                badpath=Path(tmp)/'bad.json';badpath.write_text(json.dumps(bad))
                OptimizedSourceRollout(f,initial,G1_JOINT_NAMES,source,.002,
                    source.new_tensor(unitree_armatures(G1_JOINT_NAMES)),effort,badpath,report)


class ProjectionProbeGateTests(unittest.TestCase):
    def test_only_nominal_torque_failure_can_enter_opt_in_simulation(self):
        from fabrics_sim.reference.optimized_source_rollout import eligible_for_projection_probe
        checks = dict.fromkeys(('velocity','torque','wrench','collision','boundary','landing','swing_ground'), True)
        checks['torque'] = False
        self.assertFalse(eligible_for_projection_probe(checks))
        self.assertTrue(eligible_for_projection_probe(checks, True))
        self.assertFalse(checks['torque'])
        for key in ('velocity','wrench','collision','boundary','landing','swing_ground'):
            self.assertFalse(eligible_for_projection_probe(dict(checks, **{key: False}), True))
        with self.assertRaises(ValueError):
            eligible_for_projection_probe({'torque': False}, True)
