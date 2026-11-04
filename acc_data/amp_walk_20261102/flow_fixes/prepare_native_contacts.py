"""Infer source phases from the verified G1 native sole geometry, without debounce delay.
These remain geometric estimates; live ground forces alone establish support.
"""
from pathlib import Path
import pickle,json
import numpy as np
import torch,warp as wp
from test_g1 import G1PoseFabric,reference_feet_transforms,ground_motion_reference,_quat_xyzw_to_matrix
from test_g1_two_tasks import load_motion_trajectory,G1_USD_PATH
from fabrics_sim.reference.bounded_steps import select_dynamic_steps
wp.init();torch.set_grad_enabled(False)
p=Path('acc_data/amp_walk_20261102');source=p/'amp_walk_tracking.pkl'
f=G1PoseFabric(batch_size=1,device='cpu',timestep=.002,graph_capturable=False,robot_name=G1_USD_PATH.stem,task_profile='dynamics-aware')
q,fps=load_motion_trajectory(source);q,shift=ground_motion_reference(f,q,0)
centers=torch.tensor([[-.05,-.025,-.03],[-.05,.025,-.03],[.12,-.03,-.03],[.12,.03,-.03],[.14,0,-.028]])
box=torch.tensor([.035,0,-.016]);half=torch.tensor([.075,.03,.0075]);low=[]
for row in q:
 feet=reference_feet_transforms(f,row[None]);r=_quat_xyzw_to_matrix(feet[...,3:])[0];pos=feet[0,:,:3]
 sphere=pos[:,None]+torch.einsum('fij,kj->fki',r,centers)
 low.append(torch.minimum(sphere[:,:,2].min(1).values-.005,
     pos[:,2]+(r@box)[:,2]-(r[:,2].abs()*half).sum(1)).numpy())
low=np.asarray(low);contacts=low<=.001
ambiguous=~contacts.any(axis=1)
contacts[ambiguous,low[ambiguous].argmin(axis=1)]=True
modes=contacts[:,0].astype(int)+2*contacts[:,1]
# Remove only brief double-support grazes between identical single-support
# phases. Offline cleanup preserves the onset of every retained landing.
starts=np.r_[0,np.flatnonzero(np.diff(modes))+1];ends=np.r_[starts[1:],len(modes)]
suppressed=[]
for i in range(1,len(starts)-1):
    start,end=int(starts[i]),int(ends[i])
    if modes[start]==3 and end-start<6 and modes[start-1]==modes[end] and modes[end] in (1,2):
        suppressed.append([start,end])
        modes[start:end]=modes[end]
contacts=np.stack(((modes&1)!=0,(modes&2)!=0),axis=1)
with source.open('rb') as handle:data=pickle.load(handle)
data['contact_sequence']=contacts;data['contact_names']=['left_foot','right_foot']
data['contact_provenance']='Geometric inference from native G1 sole surface height <=1mm; no causal debounce delay; not measured ground truth'
output=p/'amp_walk_native_contacts.pkl'
with output.open('wb') as handle:pickle.dump(data,handle)
report=dict(source=str(source),output=str(output),method=data['contact_provenance'],ground_shift=shift,
            ambiguous_frames=int(ambiguous.sum()),suppressed_brief_grazes=suppressed,first_four_events=select_dynamic_steps(modes,4))
(p/'flow_fixes/native_contacts.json').write_text(json.dumps(report,indent=2)+'\n')
np.savez_compressed(p/'flow_fixes/native_contacts.npz',modes=modes,sole_height=low,ambiguous=ambiguous,fps=fps)
print(json.dumps(report,indent=2))
