import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from telemetry import parse_tegrastats


class TelemetryTests(unittest.TestCase):
    def test_tegrastats_gpu_memory_and_temperature_are_parsed(self):
        line = (
            "RAM 1850/3599MB SWAP 200/18183MB "
            "CPU [20%@1020,15%@1020] GR3D_FREQ 73% GPU@46.5C "
            "CPU@44C SOC2@43.5C"
        )
        result = parse_tegrastats(line)
        self.assertEqual(result["ram_used_mb"], 1850)
        self.assertEqual(result["ram_total_mb"], 3599)
        self.assertEqual(result["gpu_util_percent"], 73)
        self.assertEqual(result["gpu_temp_c"], 46.5)
        self.assertEqual(result["cpu_temp_c"], 44.0)
        self.assertEqual(result["soc_temp_c"], 43.5)

    def test_unknown_fields_stay_unknown(self):
        result = parse_tegrastats("RAM 100/200MB")
        self.assertIsNone(result["gpu_util_percent"])
        self.assertIsNone(result["gpu_temp_c"])


if __name__ == "__main__":
    unittest.main()
