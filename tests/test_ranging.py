import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from ranging import (DepthCalibration, calibrate, estimate_object_distance,
                     focal_pixels)


class RangingTests(unittest.TestCase):
    def person(self, bbox):
        return {"label": "person", "bbox": bbox}

    def test_class_prior_returns_distance_and_uncertainty(self):
        value = estimate_object_distance(
            self.person([500, 200, 780, 600]), 1280, 720, 68
        )
        self.assertEqual(value["method"], "class_prior")
        self.assertGreater(value["approx_m"], 0.4)
        self.assertLess(value["range_m"][0], value["approx_m"])
        self.assertGreater(value["range_m"][1], value["approx_m"])

    def test_clipped_person_is_not_given_a_false_precise_distance(self):
        value = estimate_object_distance(
            self.person([0, 0, 1200, 719]), 1280, 720, 68
        )
        self.assertIsNone(value["approx_m"])
        self.assertEqual(value["reason"], "target_clipped_at_frame_edge")

    def test_unknown_class_is_not_ranged(self):
        value = estimate_object_distance(
            {"label": "glass wall", "bbox": [400, 100, 800, 600]},
            1280, 720, 68,
        )
        self.assertIsNone(value["approx_m"])
        self.assertEqual(value["reason"], "no_reliable_size_prior")

    def test_calibration_persists_and_converts_relative_depth(self):
        raw = np.full((256, 256), 100.0, np.float32)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "calibration.json"
            payload = calibrate(raw, 2.0, path)
            loaded = DepthCalibration.load(path)
            self.assertEqual(payload["reference_distance_m"], 2.0)
            result = loaded.distances([{"region": "CENTER", "raw_disparity": 50.0}])
            self.assertEqual(result[0]["method"], "calibrated_depth")
            self.assertAlmostEqual(result[0]["approx_m"], 4.0)
            self.assertEqual(loaded.reference_distance_m, 2.0)

    def test_focal_length_has_expected_endpoint_behaviour(self):
        self.assertAlmostEqual(
            focal_pixels(1280, 90), 640, places=5
        )
        with self.assertRaises(ValueError):
            focal_pixels(1280, 200)


if __name__ == "__main__":
    unittest.main()

