"""Load a prepared contact intention; measured contacts and source audit are separate."""
import hashlib,numpy as np

def load_prepared_contact_plan(path,reference_path,original_modes):
    original=np.asarray(original_modes)
    with np.load(path,allow_pickle=False) as plan:
        if str(plan['reference_file_sha256'].item())!=hashlib.sha256(reference_path.read_bytes()).hexdigest():
            raise ValueError('Prepared contact plan/reference mismatch')
        contacts=plan['contacts']
        if contacts.shape!=(len(original),2) or not np.isin(contacts,[False,True]).all():
            raise ValueError('Prepared contact plan must contain two binary feet per source frame')
        modes=contacts[:,0].astype(np.int64)+2*contacts[:,1].astype(np.int64)
        if np.any(modes==0):raise ValueError('Prepared walking contact plan cannot introduce unsupported flight')
    changed=np.flatnonzero(modes!=original)
    return modes,dict(reference=str(reference_path),reference_file_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),changed_source_frames=changed.tolist(),original_modes=original[changed].tolist(),prepared_modes=modes[changed].tolist(),scope='Controller contact intentions only; measured ground-force gating, source step events and original PKL audit unchanged')
