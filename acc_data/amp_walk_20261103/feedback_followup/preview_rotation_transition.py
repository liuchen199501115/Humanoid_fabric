"""Linearize the actual semi-implicit quaternion transition in error coordinates."""
import numpy as np
from scipy.spatial.transform import Rotation
from preview_integration import substeps

def rotation_step(error,velocity_error,acceleration,reference_quaternion,reference_velocity,next_reference_quaternion,duration):
 rotation=Rotation.from_rotvec(error)*Rotation.from_quat(reference_quaternion)
 omega=np.asarray(reference_velocity)+np.asarray(velocity_error)
 for step in substeps(duration):
  omega=omega+step*acceleration
  rotation=Rotation.from_rotvec(step*omega)*rotation
 return (rotation*Rotation.from_quat(next_reference_quaternion).inv()).as_rotvec()

def linearize_rotation(error,velocity_error,acceleration,reference_quaternion,reference_velocity,next_reference_quaternion,duration,epsilon=1e-5):
 values=[np.asarray(x,dtype=float) for x in (error,velocity_error,acceleration)]
 def evaluate(args):return rotation_step(*args,reference_quaternion,reference_velocity,next_reference_quaternion,duration)
 nominal=evaluate(values);jacobians=[]
 for block in range(3):
  J=np.zeros((3,3))
  for axis in range(3):
   plus=[x.copy() for x in values];minus=[x.copy() for x in values]
   plus[block][axis]+=epsilon;minus[block][axis]-=epsilon
   J[:,axis]=(evaluate(plus)-evaluate(minus))/(2*epsilon)
  jacobians.append(J)
 offset=nominal-sum(J@x for J,x in zip(jacobians,values))
 return offset,*jacobians
