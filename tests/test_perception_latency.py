import unittest

from perception.latency import live_capture_age, select_capture_age


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

    def test_live_capture_age_converts_server_clock_to_client_clock(self):
        # Server is 4 ms ahead. A server capture at 10.100 corresponds to
        # client time 10.096, received by the client at 10.236.
        self.assertAlmostEqual(
            live_capture_age(10.236, 10.100, 0.004), 0.140, places=6
        )

    def test_live_capture_age_rejects_invalid_or_stale_values(self):
        self.assertIsNone(live_capture_age(10.0, None, 0.0))
        self.assertIsNone(live_capture_age(10.0, 11.0, 0.0))
        self.assertIsNone(live_capture_age(10.0, 7.0, 0.0))


if __name__ == "__main__":
    unittest.main()
