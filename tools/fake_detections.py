"""Virtual objects for a robot run without objects or camera (robot PC, ROS).

Publishes ``/camera_debug/detections`` (the camera_debug JSON snapshot) for
virtual objects that ride the REAL belt: each object's y follows the encoder
(``/conveyor/distance_mm``), so the app's dead reckoning and these detections
agree. Objects are reported only inside the real camera's view band
(y 2.15..2.79 m), as the camera would. Run it INSTEAD of camera_debug
(``GP8_DEMO_NO_CAMERA=1 gp8_cycle``) once the app logs "arm rest":

    source scripts/env.sh
    python3 -m gp8_control.tools.fake_detections transparent,metal 16 1.4

positional: CLASSES (comma list, cycled in order), N, INTERVAL_S [, START_DELAY_S].
Lanes x = 0.39 / 0.45 / 0.51 m in turn; footprint 0.07 x 0.15 m. Exits when the
last object has left the view. The suction fires on an empty belt and pushes
hit nothing: this tests the decision / planning / command path.
"""
import json
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, String

from gp8_control.perception import extrinsics
from gp8_control.perception.bbox_geometry import bbox_to_base

CLASSES = sys.argv[1].split(",")
N = int(sys.argv[2])
INTERVAL = float(sys.argv[3])
DELAY = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
Y_SPAWN, Y_VIEW_LO = 2.79, 2.15
LANES = (0.39, 0.45, 0.51)
HX, HY = 0.035, 0.075            # footprint half sizes [m]
HZ = 12.0


class Fake(Node):
    def __init__(self):
        super().__init__("fake_dets")
        self.pub = self.create_publisher(String, "/camera_debug/detections", 10)
        self.dist = None
        self.v = 0.12
        self.create_subscription(Float64, "/conveyor/distance_mm", self._d, 10)
        self.create_subscription(Float64, "/conveyor/speed", self._v, 10)
        self.objs = []            # [id, cls, x, d_spawn]
        self.t0 = time.time() + DELAY
        self.n = 0
        self.create_timer(1.0 / HZ, self._tick)

    def _d(self, m):
        self.dist = float(m.data) / 1000.0

    def _v(self, m):
        self.v = float(m.data)

    def _det(self, cls, x, y, oid):
        ref_x, ref_y, ref_z = (extrinsics.REFERENCE_X_BASE, extrinsics.REFERENCE_Y_BASE,
                               extrinsics.REFERENCE_Z_BASE)
        sx = extrinsics.SIGN_CX_TO_BASE_X * extrinsics.SCALE_CX_TO_BASE_X
        sy = extrinsics.SIGN_CY_TO_BASE_Y * extrinsics.SCALE_CY_TO_BASE_Y
        og, oa = extrinsics.DETECTION_OFFSET_GRASP, extrinsics.DETECTION_OFFSET_AIM
        cx, cy = (x - ref_x) / sx, (y - ref_y) / sy
        corners = [(x - HX, y - HY), (x - HX, y + HY), (x + HX, y + HY), (x + HX, y - HY)]
        cam_bbox = np.array([[(a - ref_x) / sx, (b - ref_y) / sy, 0.0] for a, b in corners])
        kw = dict(ref_x=ref_x, ref_y=ref_y, ref_z=ref_z, scale_x=sx, scale_y=sy,
                  y_back_projection=0.0)
        return {"class": cls, "confidence": 0.9, "cam": [cx, cy, 0.0],
                "cam_bbox": cam_bbox.tolist(),
                "base_grasp": [x, y, ref_z + og], "base_aim": [x, y, ref_z + oa],
                "base_bbox_grasp": bbox_to_base(cam_bbox, z_offset=og, **kw).tolist(),
                "base_bbox_aim": bbox_to_base(cam_bbox, z_offset=oa, **kw).tolist(),
                "in_workspace": bool(abs(cx) < extrinsics.WORKSPACE_X_ABS),
                "fake_id": oid}

    def _tick(self):
        if self.dist is None:
            return
        now = time.time()
        while self.n < N and now >= self.t0 + self.n * INTERVAL:
            cls = CLASSES[self.n % len(CLASSES)]
            x = LANES[self.n % len(LANES)]
            self.objs.append([self.n + 1, cls, x, self.dist])
            self.get_logger().info(f"virtual object {self.n + 1}: {cls} x={x:.2f} enters view")
            self.n += 1
        dets = []
        for oid, cls, x, d0 in self.objs:
            y = Y_SPAWN - (self.dist - d0)
            if Y_VIEW_LO <= y <= Y_SPAWN:
                dets.append(self._det(cls, x, y, oid))
        self.pub.publish(String(data=json.dumps(
            {"receipt_time": now, "belt_mps": self.v, "detections": dets, "fake": True})))
        if self.n >= N and not dets and self.objs and now > self.t0 + N * INTERVAL + 2:
            self.get_logger().info("all virtual objects have left the view; done")
            raise SystemExit(0)


def main():
    rclpy.init()
    node = Fake()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
