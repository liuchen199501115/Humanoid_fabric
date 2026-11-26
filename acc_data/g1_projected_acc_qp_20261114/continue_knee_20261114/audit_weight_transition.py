"""Check that delayed costs preserve the entire measured prefix and single solve path."""
import json
from pathlib import Path
import numpy as np
from g1_acceleration_cost import linear_acceleration_weight
root=Path(__file__).resolve().parent
fields=('history_time','history_q','history_qd','history_qdd_qp','history_qdd_qp_reference','servo_torque')
rows=[]
with np.load(root/'baseline_traced.failure.npz') as baseline:
 for name in ('linear300_late','linear1000_late','linear300_later','baseline_final'):
  if not (root/(name+'.result.json')).exists():continue
  path=root/(name+'.npz')
  if not path.exists():path=root/(name+'.failure.npz')
  if not path.exists():continue
  with np.load(path) as a:
   options=json.loads(str(a['controller_options']))
   t=a['history_time'];start=options['qp_base_linear_weight_start']
   stop=float(t[-1]) if name=='baseline_final' else start
   for field in fields:
    times=a['servo_time'] if field=='servo_torque' else t
    bt=baseline['servo_time'] if field=='servo_torque' else baseline['history_time']
    np.testing.assert_array_equal(a[field][times<=stop],baseline[field][bt<=stop])
   expected=np.array([linear_acceleration_weight(options['qp_base_acceleration_weight'],
       options['qp_base_linear_acceleration_weight'],float(time),start=start,
       ramp=options['qp_base_linear_weight_ramp']) for time in t],dtype=np.float32)
   np.testing.assert_array_equal(a['history_qp_base_linear_acceleration_weight'],expected)
   assert np.all(a['history_qp_native_solve_count']==1)
   np.testing.assert_array_equal(a['history_qdd_qp_reference'],a['history_qdd_projected'])
   rows.append(dict(name=name,bitwise_equal_prefix_through_simulation_s=stop,
                   equal_fields=fields,recorded_cost_matches_schedule=True,
                   one_native_qp_per_tick=True,reference_is_projected=True,
                   peak_linear_cost=float(expected.max())))
(root/'weight_transition_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
