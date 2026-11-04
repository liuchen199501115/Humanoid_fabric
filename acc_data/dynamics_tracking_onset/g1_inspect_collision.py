from isaacsim import SimulationApp
app=SimulationApp({'headless':True})
from pxr import Usd, UsdGeom, UsdPhysics
import sys
sys.path.insert(0,'examples')
from test_g1_two_tasks import G1_USD_PATH
stage=Usd.Stage.Open(str(G1_USD_PATH))
for p in stage.Traverse(Usd.TraverseInstanceProxies()):
 if 'ankle_roll_link/collisions' in str(p.GetPath()):
  print('COLLIDER',p.GetPath(),p.GetTypeName(), flush=True)
  for a in p.GetAttributes():
   if a.HasAuthoredValueOpinion(): print(a.GetName(),str(a.Get())[:1800],flush=True)
app.close()
