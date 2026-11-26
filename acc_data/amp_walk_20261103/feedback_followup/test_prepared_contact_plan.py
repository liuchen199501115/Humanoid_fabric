from pathlib import Path
import tempfile,hashlib,unittest,numpy as np
from prepared_contact_plan import load_prepared_contact_plan
class PreparedContactTests(unittest.TestCase):
    def test_contact_intentions_are_bound_to_reference_and_do_not_mutate_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);ref=p/'reference.npz';ref.write_bytes(b'reference');original=np.array([1,2,2,2]);before=original.copy()
            np.savez(p/'plan.npz',contacts=[[1,0],[1,1],[1,1],[0,1]],reference_file_sha256=hashlib.sha256(ref.read_bytes()).hexdigest())
            result,report=load_prepared_contact_plan(p/'plan.npz',ref,original)
            np.testing.assert_array_equal(original,before);np.testing.assert_array_equal(result,[1,3,3,2]);self.assertEqual(report['changed_source_frames'],[1,2])
            ref.write_bytes(b'different reference')
            with self.assertRaisesRegex(ValueError,'mismatch'):load_prepared_contact_plan(p/'plan.npz',ref,original)
    def test_rejects_nonbinary_and_unsupported_contact_intentions(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);ref=p/'reference.npz';ref.write_bytes(b'reference')
            for contacts in [[[0,0]],[[2,0]],[[1,0],[0,1]]]:
                np.savez(p/'plan.npz',contacts=contacts,reference_file_sha256=hashlib.sha256(ref.read_bytes()).hexdigest())
                with self.assertRaises(ValueError):load_prepared_contact_plan(p/'plan.npz',ref,np.array([1]))
if __name__=='__main__':unittest.main()
