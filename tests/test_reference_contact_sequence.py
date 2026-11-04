import unittest

import numpy as np

from fabrics_sim.reference.manager import contact_sequence_modes


class ContactSequenceTests(unittest.TestCase):
    def test_preserves_short_support_phases_and_foot_order(self):
        contacts = np.array([[1, 1], [1, 0], [0, 1], [1, 1]], dtype=bool)
        original = contacts.copy()
        np.testing.assert_array_equal(contact_sequence_modes(contacts, 4), [3, 1, 2, 3])
        np.testing.assert_array_equal(contacts, original)

    def test_flight_is_rejected_not_replaced_by_a_support_foot(self):
        with self.assertRaisesRegex(ValueError, 'flight'):
            contact_sequence_modes([[1, 1], [0, 0]], 2)

    def test_invalid_length_and_labels(self):
        for contacts, count in (([[1, 0]], 2), ([[2, 0]], 1),
                                ([[float('nan'), 1]], 1), ([], 0)):
            with self.subTest(contacts=contacts):
                with self.assertRaises(ValueError):
                    contact_sequence_modes(contacts, count)


if __name__ == '__main__':
    unittest.main()
