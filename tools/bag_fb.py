"""Encoder feedback from a rosbag into a run recording (joints_fb.f64).

    python3 -m gp8_control.tools.bag_fb RECORD_DIR      # system ROS python (rosbag2_py)

``scripts/cycle_demo.sh`` records ``/dynamic_joint_states`` with the C++
``ros2 bag record`` into ``RECORD_DIR/fb_bag`` (a subscription inside the
Python app cost ~20 % of a core at 500 Hz) and runs this when the run ends.
Rows are ``[monotonic, q1..q6]`` of ``pos_fdb`` (the encoder; the driver's
``/joint_states`` position is the accepted set point ``pos_set``), timed by
the joint_state_broadcaster's header stamp mapped onto the app's monotonic
clock with the recording's own wall/monotonic pair (meta event).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

JOINT_NAMES = ["joint_1_s", "joint_2_l", "joint_3_u", "joint_4_r", "joint_5_b", "joint_6_t"]


def _wall_minus_mono(rec_dir: Path) -> float:
    with open(rec_dir / "events.jsonl", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if e.get("ev") == "meta":
                return float(e["w"]) - float(e["m"])
    raise SystemExit("no meta event in the recording")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    rec_dir = Path(argv[0])
    bag = rec_dir / "fb_bag"
    if not bag.exists():
        print(f"[bag_fb] no {bag}; nothing to convert")
        return 0
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    off = _wall_minus_mono(rec_dir)
    reader = rosbag2_py.SequentialReader()
    storage = rosbag2_py.StorageOptions(uri=str(bag), storage_id="")
    reader.open(storage, rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    cls = get_message(types["/dynamic_joint_states"])
    rows = []
    while reader.has_next():
        topic, data, t_bag = reader.read_next()
        if topic != "/dynamic_joint_states":
            continue
        m = deserialize_message(data, cls)
        vals = {}
        for name, iv in zip(m.joint_names, m.interface_values):
            for k, v in zip(iv.interface_names, iv.values):
                if k == "pos_fdb":
                    vals[name] = v
        if not all(j in vals for j in JOINT_NAMES):
            continue
        t_wall = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec
        if t_wall <= 0.0:
            t_wall = t_bag * 1e-9
        rows.append([t_wall - off, *[vals[j] for j in JOINT_NAMES]])
    if not rows:
        print("[bag_fb] no pos_fdb samples in the bag")
        return 1
    a = np.asarray(rows, "<f8")
    a = a[np.argsort(a[:, 0], kind="stable")]
    a.tofile(rec_dir / "joints_fb.f64")
    dt = np.diff(a[:, 0])
    print(f"[bag_fb] {len(a)} encoder samples, {1.0 / np.median(dt):.0f} Hz -> {rec_dir / 'joints_fb.f64'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
