"""Smoke checks for object-resolved RL reward bookkeeping."""

from __future__ import annotations

import time

import numpy as np

from gp8_control.backends.mujoco_sim import (
    class_bin_name_for,
    resolved_object_reward,
)
from gp8_control.perception.detection_intake import DetectionIntake
from gp8_control.tracking import TrackedObjectQueue


def _snapshot(sim_object_id: int, cls: str = "metal", x: float = 0.45) -> dict:
    return {
        "receipt_time": time.time(),
        "detections": [
            {
                "class": cls,
                "confidence": 0.9,
                "sim_object_id": sim_object_id,
                "cam": [0.0, 0.0, 0.0],
                "cam_bbox": [[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.1]],
                "base_grasp": [x, 0.3, 0.042],
                "base_aim": [x, 0.3, 0.10],
                "base_bbox_grasp": [[x, 0.3], [x + 0.1, 0.3]],
                "base_bbox_aim": [[x, 0.3], [x + 0.1, 0.3]],
                "in_workspace": True,
            }
        ],
    }


def main(argv=None) -> int:
    del argv
    checks = [
        (
            "manipulated metal in class bin",
            resolved_object_reward(
                manipulated=True,
                actual_bin_name="push_metal",
                class_bin_name="push_metal",
                collateral=False,
            ) == 1.0,
        ),
        (
            "manipulated metal wrong bin",
            resolved_object_reward(
                manipulated=True,
                actual_bin_name="throw",
                class_bin_name="push_metal",
                collateral=False,
            ) == -0.3,
        ),
        (
            "manipulated transparent in class bin",
            resolved_object_reward(
                manipulated=True,
                actual_bin_name="throw",
                class_bin_name="throw",
                collateral=False,
            ) == 1.0,
        ),
        (
            "manipulated transparent wrong bin",
            resolved_object_reward(
                manipulated=True,
                actual_bin_name="push_metal",
                class_bin_name="throw",
                collateral=False,
            ) == -0.3,
        ),
        (
            "manipulated outside all bins",
            resolved_object_reward(
                manipulated=True,
                actual_bin_name=None,
                class_bin_name="throw",
                collateral=False,
            ) == -0.3,
        ),
        (
            "unmanipulated collateral",
            resolved_object_reward(
                manipulated=False,
                actual_bin_name=None,
                class_bin_name="throw",
                collateral=True,
            ) == -0.3,
        ),
        (
            "unmanipulated natural exit",
            resolved_object_reward(
                manipulated=False,
                actual_bin_name=None,
                class_bin_name="throw",
                collateral=False,
            ) == 0.0,
        ),
        ("class bin metal", class_bin_name_for("metal") == "push_metal"),
        ("class bin transparent", class_bin_name_for("transparent") == "throw"),
        ("class bin unknown", class_bin_name_for("paper") is None),
    ]

    queue = TrackedObjectQueue(max_reach=1.0, drop_below_y=-1.0)
    intake = DetectionIntake(eps=0.05)
    intake.ingest(_snapshot(101), queue, None, v=0.1, logger=None)
    first = queue._objects[0]
    checks.append(("new track carries sim id", first.sim_object_id == 101))
    later = _snapshot(202)
    later["receipt_time"] += 0.01
    intake.ingest(later, queue, None, v=0.1, logger=None)
    checks.extend(
        [
            ("matched redetection keeps one track", len(queue._objects) == 1),
            ("matched redetection updates sim id", queue._objects[0].sim_object_id == 202),
        ]
    )

    failures = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if failures:
        print("FAILURES:", ", ".join(failures))
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
