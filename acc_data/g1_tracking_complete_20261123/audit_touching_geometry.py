"""Replay observed manifold support selection, without changing simulation history."""
import json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_leg_kinematics import LegKinematics
from fabrics_sim.solver.observed_sole_contact import ObservedSoleContact
name=sys.argv[1] if len(sys.argv)>1 else 'eleven_pause960_search2'
a=np.load(OUT/(name+'.npz'));legs=LegKinematics(str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'),a['final_joint_order'].tolist());rows=[]
for time in [4.23,4.3,4.35,4.4,4.45,4.5]:
 i=int(np.argmin(abs(a['history_time']-time)));start=int(a['history_contact_start'][i,0,0]);count=int(a['history_contact_count'][i,0,0]);sl=slice(start,start+count)
 points=a['history_contact_point'][i,sl];forces=a['history_contact_normal_force'][i,sl].reshape(-1);gaps=a['history_contact_distance'][i,sl].reshape(-1)
 position,rotation,_=legs.evaluate(a['history_q'][i,0],0);position=torch.tensor(position[None]);rotation=torch.tensor(rotation[None]);variants=[]
 for tolerance in [None,1e-5,5e-5,1e-4]:
  patch=ObservedSoleContact(position,rotation,0.);patch.flatten=True;patch.update_support(position,rotation,points,forces,gaps,touching_tolerance=tolerance)
  variants.append(dict(tolerance_m=tolerance,kind='polygon' if patch.polygon_support else 'point' if patch.point_support else 'line',vertices=patch.vertices[0].tolist(),normal_load_n=float(forces.sum())))
 rows.append(dict(time=float(a['history_time'][i]),reported_points=points.tolist(),measured_loads_n=forces.tolist(),separations_m=gaps.tolist(),variants=variants))
(OUT/(name+'.touching_geometry.json')).write_text(json.dumps(dict(source_trial=name,offline_geometry_only=True,rows=rows),indent=2)+'\n')
for row in rows: print(row['time'],[(v['tolerance_m'],v['kind'],len(v['vertices'])) for v in row['variants']])
