import json
import unittest
from unittest.mock import patch

from perception.perception_client import stream_detections


class _Response:
    def __init__(self, records):
        self._lines = [(json.dumps(record) + "\n").encode() for record in records]

    def __enter__(self):
        return iter(self._lines)

    def __exit__(self, *_args):
        return False


class _Stop(Exception):
    pass


class TestPerceptionClient(unittest.TestCase):
    def test_can_skip_cached_initial_record(self):
        received = []

        def on_record(record):
            received.append(record)
            raise _Stop

        response = _Response([
            {"schema_version": 2, "timestamp": 1.0},
            {"schema_version": 2, "timestamp": 2.0},
        ])
        with patch("perception.perception_client.urllib.request.urlopen", return_value=response):
            with self.assertRaises(_Stop):
                stream_detections(
                    "http://camera/detections/stream",
                    on_record,
                    skip_initial_record=True,
                )
        self.assertEqual(received, [{"schema_version": 2, "timestamp": 2.0}])


if __name__ == "__main__":
    unittest.main()
