import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
from g1_soft_joint_braking import soft_joint_braking_rows
from g1_soft_tracking_tube import SoftTrackingTubeBackend
from fabrics_sim.solver.admm_qp import QPData

class SoftBrakingTests(unittest.TestCase):
    def test_braking_direction_and_departure(self):
        q=torch.zeros(1,29,dtype=torch.float64);v=q.clone();lo=q-1.;hi=q+1.
        q[0,13]=.8;v[0,13]=2.
        C,d=soft_joint_braking_rows(q,v,lo,hi,[13])
        self.assertEqual(float(C[0,1,19]),1.)
        self.assertLess(float(d[0,1]),0.)
        self.assertFalse(C[0,0].any())
        v[0,13]=-2.
        C,d=soft_joint_braking_rows(q,v,lo,hi,[13])
        self.assertFalse(C.any())
        self.assertFalse(d.any())

    def test_one_solve_and_no_penalty_for_stronger_braking(self):
        H=torch.eye(35,dtype=torch.float64)[None];h=torch.zeros(1,35,dtype=torch.float64)
        qp=QPData(H,h,H.new_zeros(1,0,35),h.new_zeros(1,0),h.new_full(h.shape,-100.),h.new_full(h.shape,100.))
        solver=SoftTrackingTubeBackend(3.)
        solver.tube=(H.new_zeros(1,3,35),h.new_zeros(1,3),h.new_ones(3))
        C=H.new_zeros(1,1,35);C[0,0,19]=1.
        solver.braking=(C,h.new_tensor([[-5.]]),3.)
        # Target -10 satisfies a<=-5: an equality spring would incorrectly pull it back.
        qp.h[0,19]=10.
        result=solver.solve(qp)
        self.assertTrue(result.status_ok.all())
        self.assertAlmostEqual(float(result.z[0,19]),-10.,places=7)
        self.assertAlmostEqual(float(solver.last_braking_slack[0,0]),0.,places=7)
        self.assertEqual(solver.last_solve_counts,[1])
        self.assertEqual(solver.last_slack.shape,(1,3))
        qp.h[0,19]=-2.
        result=solver.solve(qp)
        # min .5*(a-2)^2+1.5*(a+5)^2 gives a=-3.25, slack=1.75.
        self.assertTrue(result.status_ok.all())
        self.assertAlmostEqual(float(result.z[0,19]),-3.25,places=6)
        self.assertAlmostEqual(float(solver.last_braking_slack[0,0]),1.75,places=6)
        self.assertEqual(solver.last_solve_counts,[1])
        qp.A_eq=H[:,19:20].clone();qp.b_eq=h.new_tensor([[0.]])
        result=solver.solve(qp)
        self.assertTrue(result.status_ok.all())
        self.assertAlmostEqual(float(result.z[0,19]),0.,places=6)
        self.assertFalse(solver.last_augmented_problem.A_eq[:,:,35:].any())

    def test_invalid_joint_indices_rejected(self):
        q=torch.zeros(1,29)
        for indices in ([29],[13,13],[]):
            with self.assertRaises(ValueError):soft_joint_braking_rows(q,q,q-1,q+1,indices)

if __name__=='__main__':unittest.main()
