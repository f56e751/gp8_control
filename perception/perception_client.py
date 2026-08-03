"""Reusable client for the perception detection stream.

COPY THIS FILE INTO YOUR ROBOT REPO. It depends only on the Python stdlib
(no torch / pyrealsense2 / this repo), so the robot PC consumes the perception
output purely through the documented wire contract:

    GET http://<camera-pc-ip>:8080/detections/stream   (NDJSON, one record/line)
    {
      "schema_version": 2,
      "timestamp": <epoch s>, "elapsed_s": <inference s>,
      "capture_timestamp": <RealSense global-time epoch s or null>,
      "capture_age_s": <frame-to-inference-start seconds or null>,
      "capture_timestamp_domain": <RealSense timestamp domain>,
      "bounding_boxes": [                # belt frame, metres; clockwise
        [[X_tl,Y_tl,Z], [X_tr,Y_tr,Z], [X_br,Y_br,Z], [X_bl,Y_bl,Z]], ...
      ],
      "class_names": ["metal"|"transparent"|"cardboard", ...],
      "confidences": [<float>, ...]      # parallel arrays, aligned by index
    }

stream_detections() reconnects automatically on network drops; exceptions from
your on_record callback propagate (so robot-side bugs surface instead of being
silently retried).
"""
import json
import time
import urllib.error
import urllib.request

# Schema this client was written against. A mismatch is warned about once.
EXPECTED_SCHEMA_VERSION = 2


def stream_detections(url, on_record, reconnect_delay=2.0, verify_schema=True):
    """Connect to the NDJSON stream and call on_record(dict) for each record.

    Loops forever, reconnecting after `reconnect_delay` seconds whenever the
    connection drops (camera PC restart, LAN blip, ...).
    """
    warned = False
    while True:
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                for raw in resp:
                    line = raw.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if verify_schema and not warned:
                        version = record.get("schema_version")
                        if version != EXPECTED_SCHEMA_VERSION:
                            print(
                                f"[perception] WARNING: stream schema_version "
                                f"{version} != expected {EXPECTED_SCHEMA_VERSION}; "
                                f"fields may have changed.",
                                flush=True,
                            )
                            warned = True
                    on_record(record)
        except (urllib.error.URLError, OSError) as e:
            print(
                f"[perception] disconnected: {e}; retrying in {reconnect_delay}s",
                flush=True,
            )
            time.sleep(reconnect_delay)


def _demo(record):
    for box, cls, conf in zip(
        record["bounding_boxes"], record["class_names"], record["confidences"]
    ):
        points = " ".join(f"({x:.3f},{y:.3f},{z:.3f})" for x, y, z in box)
        print(f"  {cls:11s} conf={conf:.2f}  bbox={points} m")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--url",
        default="http://147.46.175.15:8080/detections/stream",
        help="NDJSON detection stream URL exposed by the camera PC's main.py.",
    )
    args = ap.parse_args()
    print(f"connecting to {args.url} ...", flush=True)
    stream_detections(args.url, _demo)
