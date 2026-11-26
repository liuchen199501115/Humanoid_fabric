import unittest
import numpy as np
from anchored_preview_grid import anchored_grid,anchored_control_grid
class AnchoredGridTests(unittest.TestCase):
 def test_distant_coarse_nodes_remain_fixed(self):
  modes=np.full(2000,2);a,da=anchored_grid(840.2,240.,modes,840.2);b,db=anchored_grid(842.6,240.,modes,840.2)
  shared_a=a[(a>b[2]+1e-8)&(a<min(a[-1],b[-1])-1e-8)];shared_b=b[(b>b[2]+1e-8)&(b<min(a[-1],b[-1])-1e-8)]
  np.testing.assert_allclose(shared_a,shared_b,atol=1e-10)
  self.assertTrue(np.all(da>0));self.assertTrue(np.all(db>0))
 def test_short_double_support_is_represented(self):
  modes=np.full(2000,2);modes[884:887]=3;modes[887:]=1
  frames,dt=anchored_grid(840.2,240.,modes,840.2)
  phases=modes[np.floor(frames+1e-8).astype(int)];self.assertIn(3,phases)
  self.assertTrue(np.allclose((frames[:-1]-840.2)/(.002*240.),np.round((frames[:-1]-840.2)/(.002*240.)),atol=1e-8))
 def test_original_endpoint_is_not_extended(self):
  modes=np.full(2000,2);frames,dt=anchored_grid(1997.9,240.,modes,840.2)
  self.assertEqual(frames[-1],1999.);self.assertEqual(len(frames),2);self.assertAlmostEqual(dt.sum(),1.1/240.)
 def test_invalid_domain_rejected(self):
  with self.assertRaises(ValueError):anchored_grid(1999.,240.,np.full(2000,2),840.2)
  with self.assertRaises(ValueError):anchored_grid(840.,0.,np.full(2000,2),840.2)
class DenseControlGridTests(unittest.TestCase):
 def test_all_first_twenty_controls_are_two_milliseconds(self):
  f,dt=anchored_control_grid(872.36,240.,np.full(2000,2),872.36)
  np.testing.assert_allclose(dt[:20],.002,atol=1e-12)
  self.assertTrue(np.all(dt>0));self.assertGreaterEqual(dt.sum(),.544-1e-10)
 def test_short_support_transition_survives_coarse_tail(self):
  modes=np.full(2000,1);modes[952:955]=3;modes[955:]=2
  f,dt=anchored_control_grid(872.36,240.,modes,872.36)
  phases=modes[np.floor(f+1e-8).astype(int)]
  self.assertIn(3,phases);self.assertIn(2,phases)
  entry=f[np.flatnonzero(phases==3)[0]]
  self.assertGreaterEqual(entry,952.);self.assertLess(entry,952.+.48+1e-8)
 def test_only_final_interval_can_end_between_control_ticks(self):
  f,dt=anchored_control_grid(1997.9,240.,np.full(2000,2),840.2)
  self.assertEqual(f[-1],1999.);self.assertTrue(np.all(dt>0))
  np.testing.assert_allclose(dt[:-1],.002,atol=1e-12)
  self.assertLessEqual(dt[-1],.002)
 def test_coarse_nodes_survive_a_full_replanning_period(self):
  modes=np.full(2000,2)
  a,_=anchored_control_grid(872.36,240.,modes,872.36)
  b,_=anchored_control_grid(881.96,240.,modes,872.36)
  lower=b[20]+1e-8;upper=min(a[-1],b[-1])-1e-8
  np.testing.assert_allclose(a[(a>lower)&(a<upper)],b[(b>lower)&(b<upper)],atol=1e-10)

if __name__=='__main__':unittest.main()
