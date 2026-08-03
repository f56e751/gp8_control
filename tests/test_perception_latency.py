import unittest

from perception.latency import select_capture_age


class TestPerceptionLatency(unittest.TestCase):
    def test_prefers_realsense_capture_age(self):
        self.assertEqual(
            select_capture_age(0.042, 0.100),
            (0.042, "realsense_global_time"),
        )

    def test_missing_or_invalid_value_uses_estimate(self):
        for value in (None, "bad", -0.1, float("nan"), 3.0):
            with self.subTest(value=value):
                self.assertEqual(
                    select_capture_age(value, 0.100),
                    (0.100, "estimated_frame_period"),
                )


if __name__ == "__main__":
    unittest.main()
