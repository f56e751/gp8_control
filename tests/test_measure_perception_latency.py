import unittest

from tools.measure_perception_latency import (
    ProbeSample,
    calculate_probe,
    estimate_clock_offset,
    percentile,
)


class TestLatencyMath(unittest.TestCase):
    def test_ntp_equations_remove_server_work(self):
        # Client sends at 10.000. Server clock is +0.100 s. Network is 0.010 s
        # each way and the server spends 0.005 s handling the request.
        sample = calculate_probe(10.000, 10.110, 10.115, 10.025)
        self.assertAlmostEqual(sample.rtt_s, 0.020, places=6)
        self.assertAlmostEqual(sample.offset_s, 0.100, places=6)

    def test_offset_uses_low_rtt_samples(self):
        samples = [
            ProbeSample(0.010, 0.100), ProbeSample(0.011, 0.101),
            ProbeSample(0.012, 0.099), ProbeSample(0.200, 0.180),
            ProbeSample(0.300, 0.050),
        ]
        self.assertAlmostEqual(estimate_clock_offset(samples), 0.100, places=6)

    def test_percentile_interpolates(self):
        self.assertEqual(percentile([0, 10, 20], 50), 10)
        self.assertEqual(percentile([0, 10], 95), 9.5)


if __name__ == "__main__":
    unittest.main()
