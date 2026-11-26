"""World-tangent pose operations without importing prediction solver libraries."""
import numpy as np
from scipy.spatial.transform import Rotation

def difference(q,target):
 return np.r_[q[:3]-target[:3],(Rotation.from_quat(q[3:7])*Rotation.from_quat(target[3:7]).inv()).as_rotvec(),q[7:]-target[7:]]
def integrate(q,delta):
 result=q.copy();result[:3]+=delta[:3];result[3:7]=(Rotation.from_rotvec(delta[3:6])*Rotation.from_quat(q[3:7])).as_quat();result[7:]+=delta[6:];return result
