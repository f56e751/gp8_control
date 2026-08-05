import unittest

from perception.clock_sync import (
    ProbeSample,
    calculate_probe,
    estimate_clock_offset,
    latency_url_from_stream,
)


class TestClockSync(unittest.TestCase):
    def test_ntp_equations_remove_server_work(self):
        sample = calculate_probe(10.000, 10.110, 10.115, 10.025)
        self.assertAlmostEqual(sample.rtt_s, 0.020, places=6)
        self.assertAlmostEqual(sample.offset_s, 0.100, places=6)

    def test_offset_uses_low_rtt_samples(self):
        samples = [
            ProbeSample(0.010, 0.100),
            ProbeSample(0.011, 0.101),
            ProbeSample(0.012, 0.099),
            ProbeSample(0.200, 0.180),
            ProbeSample(0.300, 0.050),
        ]
        offset, min_rtt = estimate_clock_offset(samples)
        self.assertAlmostEqual(offset, 0.100, places=6)
        self.assertEqual(min_rtt, 0.010)

    def test_latency_url_replaces_stream_path(self):
        self.assertEqual(
            latency_url_from_stream(
                "http://147.46.175.15:8080/detections/stream"
            ),
            "http://147.46.175.15:8080/latency",
        )


if __name__ == "__main__":
    unittest.main()
