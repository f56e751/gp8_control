import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from gp8_control.controllers.trajectory_controller import TrajectoryController
from gp8_control.conveyor.conveyor_speed import ConveyorSpeedTracker
from gp8_control.perception.detection_intake import DetectionIntake
from gp8_control.tracking.object_queue import TrackedObject, TrackedObjectQueue


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warn(self, message):
        self.messages.append(("warn", message))


class _Node:
    def __init__(self):
        self.callbacks = {}
        self.logger = _Logger()

    def create_subscription(self, _type, topic, callback, _qos):
        self.callbacks[topic] = callback
        return object()

    def get_logger(self):
        return self.logger


def _pose(y):
    pose = np.eye(4)
    pose[1, 3] = y
    return pose


def _snapshot(receipt_time, x, y):
    return {
        "receipt_time": receipt_time,
        "detections": [{
            "in_workspace": True,
            "class": "transparent",
            "confidence": 0.8,
            "base_grasp": [x, y, 0.62],
            "base_aim": [x, y, 0.70],
            "cam": [x, y, 0.0],
            "cam_bbox": [[x, y, 0.0]] * 4,
            "base_bbox_grasp": [[x, y, 0.62]] * 4,
            "base_bbox_aim": [[x, y, 0.70]] * 4,
        }],
    }


class TestEncoderDistanceTracking(unittest.TestCase):
    def test_tracker_interpolates_distance_and_unwraps_reset(self):
        node = _Node()
        tracker = ConveyorSpeedTracker(node, "/speed", 0.2, 2.0, "/distance")
        speed_cb = node.callbacks["/speed"]
        distance_cb = node.callbacks["/distance"]

        with patch("gp8_control.conveyor.conveyor_speed.time.time", return_value=100.0):
            speed_cb(SimpleNamespace(data=0.2))
            distance_cb(SimpleNamespace(data=1000.0))
        with patch("gp8_control.conveyor.conveyor_speed.time.time", return_value=101.0):
            distance_cb(SimpleNamespace(data=1100.0))

        self.assertAlmostEqual(tracker.distance_at(100.5), 1.05)
        with patch("gp8_control.conveyor.conveyor_speed.time.time", return_value=101.0):
            self.assertAlmostEqual(tracker.distance_m, 1.1)

        with patch("gp8_control.conveyor.conveyor_speed.time.time", return_value=101.1):
            distance_cb(SimpleNamespace(data=0.0))
        with patch("gp8_control.conveyor.conveyor_speed.time.time", return_value=101.1):
            self.assertAlmostEqual(tracker.distance_m, 1.1)
        with patch("gp8_control.conveyor.conveyor_speed.time.time", return_value=101.2):
            distance_cb(SimpleNamespace(data=20.0))
        with patch("gp8_control.conveyor.conveyor_speed.time.time", return_value=101.2):
            self.assertAlmostEqual(tracker.distance_m, 1.12)

    def test_tracked_object_uses_distance_instead_of_speed_age(self):
        obj = TrackedObject(
            T_aim_base=_pose(2.5), T_grasp_base=_pose(2.5),
            class_name="transparent", detect_time=10.0,
            encoder_distance_m=1.0,
        )
        self.assertAlmostEqual(obj.y_at(100.0, 99.0, 1.4), 2.1)

    def test_redetection_does_not_move_control_y_anchor(self):
        intake = DetectionIntake(0.12)
        tracked = TrackedObjectQueue(max_reach=4.0, drop_below_y=-4.0)

        added = intake.ingest(
            _snapshot(100.0, 0.40, 2.50), tracked, None, 0.2,
            belt_distance_m=1.0,
        )
        self.assertEqual(added, 1)
        obj = tracked.head()

        # Re-publishing the same receipt timestamp must not add votes or mutate.
        votes = dict(obj.class_votes)
        intake.ingest(
            _snapshot(100.0, 0.60, 9.00), tracked, None, 0.2,
            belt_distance_m=1.0,
        )
        self.assertEqual(obj.class_votes, votes)

        # A later camera centre is deliberately 10 cm upstream of the encoder
        # prediction. X/display data may refresh; control Y/time stay frozen.
        intake.ingest(
            _snapshot(101.0, 0.45, 2.40), tracked, None, 0.2,
            belt_distance_m=1.2,
        )
        self.assertAlmostEqual(obj.T_grasp_base[0, 3], 0.45)
        self.assertAlmostEqual(obj.T_grasp_base[1, 3], 2.50)
        self.assertEqual(obj.detect_time, 100.0)
        self.assertAlmostEqual(obj.y_at(101.0, 0.2, 1.2), 2.30)

    def test_suction_requests_coalesce_duplicate_states(self):
        ctrl = TrajectoryController.__new__(TrajectoryController)
        ctrl._io_state_lock = threading.Lock()
        ctrl._io_requested_value = None
        ctrl._io_queue = queue.Queue()
        ctrl.current_joints = None
        ctrl.last_suction_on_t = None
        ctrl.last_suction_on_joints = None
        ctrl.last_suction_off_t = None

        ctrl.suction_off()
        ctrl.suction_off()
        ctrl.suction_on()
        ctrl.suction_on()

        self.assertEqual(ctrl._io_queue.qsize(), 2)
        self.assertEqual(ctrl._io_queue.get()[1], 1)
        self.assertEqual(ctrl._io_queue.get()[1], 0)


if __name__ == "__main__":
    unittest.main()
