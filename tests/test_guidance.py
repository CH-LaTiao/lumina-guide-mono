import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from guidance import advise


def obj(label, bbox, meters):
    return {
        "label": label,
        "bbox": bbox,
        "distance": {
            "approx_m": meters,
            "range_m": [meters * .7, meters * 1.3],
            "confidence": "low",
            "method": "class_prior",
        },
    }


class GuidanceTests(unittest.TestCase):
    camera = {"state": "STREAMING", "frame_age_ms": 30}

    def test_center_near_object_chooses_the_measurably_clearer_side(self):
        result = advise(
            "walking",
            [obj("person", [450, 100, 830, 620], 1.4)],
            [
                {"region": "LEFT", "near_score": .1},
                {"region": "CENTER", "near_score": .9},
                {"region": "RIGHT", "near_score": .8},
            ],
            self.camera,
            1280,
        )
        self.assertEqual(result.action, "STOP")
        self.assertIn("左侧", result.speech)
        self.assertFalse(result.movement_allowed)

    def test_stale_camera_immediately_withdraws_route_cue(self):
        result = advise(
            "walking", [], [], {"state": "STREAMING", "frame_age_ms": 1200}, 1280
        )
        self.assertEqual(result.action, "STOP")
        self.assertEqual(result.speech, "停，感知不可用。")
        self.assertIsNone(result.direction)

    def test_right_side_object_requests_left_adjustment(self):
        result = advise(
            "walking",
            [obj("chair", [1000, 260, 1230, 600], 1.2)],
            [{"region": "CENTER", "near_score": .3}],
            self.camera,
            1280,
        )
        self.assertEqual(result.action, "ADJUST")
        self.assertIn("向左", result.speech)

    def test_pause_never_emits_a_direction(self):
        result = advise("paused", [], [], {"state": "FAULT"}, 1280)
        self.assertEqual(result.speech, "已暂停。")
        self.assertIsNone(result.direction)


if __name__ == "__main__":
    unittest.main()
