"""ROS 2 utility functions."""

from __future__ import annotations

import subprocess


def call_rosservice(service_name: str) -> None:
    """Call a ROS 2 service via CLI subprocess."""
    try:
        result = subprocess.check_output(
            ["ros2", "service", "call", service_name, "std_srvs/srv/Trigger"],
            stderr=subprocess.STDOUT,
        )
        print(f"Service '{service_name}' called successfully. Result: {result}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to call service '{service_name}'. Error: {e.output}")
