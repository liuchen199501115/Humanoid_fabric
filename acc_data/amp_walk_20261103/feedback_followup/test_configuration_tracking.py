import unittest
from types import SimpleNamespace
import torch
from fabrics_sim.fabrics.task_request import TaskRequest
from configuration_tracking import configuration_tracking_request

class ConfigurationTrackingTests(unittest.TestCase):
 def setup_case(self):
  q=torch.zeros(1,36,dtype=torch.float64);q[:,6]=1;v=torch.zeros(1,35,dtype=q.dtype);a=torch.linspace(-1,1,35,dtype=q.dtype)[None]
  reference=SimpleNamespace(q=q.clone(),v=v.clone(),a=a.clone());requests=[]
  for name,indices,weights in [('center_of_mass',range(3),250.),('source_pelvis_orientation',range(3,6),40.),('joint_posture',range(6,35),.5)]:
   indices=list(indices);A=torch.eye(35,dtype=q.dtype)[None,indices];desired=a[:,indices].clone();requests.append(TaskRequest(name,A,torch.zeros_like(desired),desired,torch.eye(len(indices),dtype=q.dtype)[None]*weights,torch.ones_like(desired,dtype=torch.bool)))
  return q,v,reference,requests
 def test_exact_reference_preserves_full_feedforward(self):
  q,v,r,tasks=self.setup_case();out=configuration_tracking_request(q,v,r,tasks)
  torch.testing.assert_close(out.desired,r.a)
  torch.testing.assert_close(out.residual(r.a),torch.zeros_like(r.a))
 def test_base_feedback_opposes_position_and_velocity_error(self):
  for velocity_error in [False,True]:
   q,v,r,tasks=self.setup_case();error=q.new_tensor([[.01,-.02,.03]])
   if velocity_error:v[:,:3]+=error
   else:q[:,:3]+=error
   feedback=configuration_tracking_request(q,v,r,tasks).desired[:,:3]-r.a[:,:3]
   self.assertTrue(bool((feedback*error<0).all()))
 def test_root_target_does_not_substitute_arm_motion_for_root_tracking(self):
  q,v,r,tasks=self.setup_case();first=configuration_tracking_request(q,v,r,tasks)
  q[:,22:]+=1;v[:,21:]+=2;tasks[-1].desired[:,15:]+=30
  second=configuration_tracking_request(q,v,r,tasks)
  torch.testing.assert_close(first.desired[:,:6],second.desired[:,:6])
  self.assertFalse(torch.equal(first.desired[:,21:],second.desired[:,21:]))
if __name__=='__main__':unittest.main()
