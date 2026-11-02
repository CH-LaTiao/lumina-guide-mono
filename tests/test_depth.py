import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from depth_estimator import normalize_inverse_depth, region_statistics


class DepthTests(unittest.TestCase):
    def test_normalization_keeps_raw_inverse_depth(self):
        raw = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)
        preserved, normalized = normalize_inverse_depth(raw[None, None, :, :])
        self.assertEqual(preserved.shape, raw.shape)
        self.assertAlmostEqual(float(normalized.min()), 0.0)
        self.assertAlmostEqual(float(normalized.max()), 1.0)

    def test_region_near_scores_follow_higher_midas_disparity(self):
        raw = np.ones((256, 256), np.float32)
        raw[:, :85] = 2
        raw[:, 85:170] = 4
        raw[:, 170:] = 8
        regions = region_statistics(raw)
        self.assertEqual([item["region"] for item in regions],
                         ["LEFT", "CENTER", "RIGHT"])
        self.assertLess(regions[0]["near_score"], regions[1]["near_score"])
        self.assertLess(regions[1]["near_score"], regions[2]["near_score"])

    def test_flat_depth_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "usable range"):
            normalize_inverse_depth(np.ones((1, 1, 256, 256), np.float32))


if __name__ == "__main__":
    unittest.main()

