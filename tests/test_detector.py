import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from detector import decode


class DetectorDecodeTests(unittest.TestCase):
    def test_person_box_is_mapped_from_letterbox_to_original_frame(self):
        output = np.zeros((1, 84, 8400), np.float32)
        output[0, :4, 0] = (320, 320, 200, 400)
        output[0, 4, 0] = 0.92
        objects = decode(output, (1.0, 0, 0, 1280, 720))
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]["label"], "person")
        self.assertEqual(objects[0]["bbox"], [220.0, 120.0, 420.0, 520.0])

    def test_non_finite_model_output_is_rejected(self):
        output = np.zeros((1, 84, 8400), np.float32)
        output[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            decode(output, (1.0, 0, 0, 1280, 720))

    def test_wrong_contract_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "contract"):
            decode(np.zeros((84, 8400), np.float32), (1.0, 0, 0, 1, 1))


if __name__ == "__main__":
    unittest.main()

