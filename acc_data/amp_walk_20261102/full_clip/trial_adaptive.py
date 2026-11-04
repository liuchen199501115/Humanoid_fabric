"""Diagnostic: increase horizontal COM weight smoothly with measured position error."""
import torch
import test_g1 as app
original = app.G1PoseFabric.compute_acceleration

def adaptive_compute(self,q,qd):
    params = self.fabric_params['center_of_mass_attractor']
    if not hasattr(self,'_trial_base_com_metric'):
        self._trial_base_com_metric = list(params['metric'])
    base = self._trial_base_com_metric
    error = float(torch.linalg.vector_norm((self.compute_center_of_mass(q)-self.center_of_mass_taskmap.position_target)[:,:2]))
    phase = min(1.,max(0.,(error-.02)/.04))
    scale = 1.+3.*phase**3*(10.-15.*phase+6.*phase**2)
    params['metric'] = [base[0]*scale,base[1]*scale,base[2]]
    return original(self,q,qd)

app.G1PoseFabric.compute_acceleration = adaptive_compute
app.main()
