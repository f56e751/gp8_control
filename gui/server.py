"""GP8 Web GUI server — FastAPI + rclpy."""
from __future__ import annotations

import asyncio
import re
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .device_monitor import DeviceMonitor
from .manual_controller import ManualController

STATIC_DIR = Path(__file__).parent / "static"

# Find project root: try symlink source first, then common locations
def _find_project_root() -> Path:
    # 1. From __file__ (works if running from source or symlink)
    candidates = [
        Path(__file__).resolve().parent.parent.parent.parent,
        Path.home() / "Documents" / "Github" / "iitp_robot_control",
    ]
    for p in candidates:
        if (p / "assets" / "urdf" / "gp8.urdf").exists():
            return p
    return candidates[-1]  # fallback

PROJECT_ROOT = _find_project_root()
URDF_PATH = PROJECT_ROOT / "assets" / "urdf" / "gp8.urdf"
MESHES_DIR = PROJECT_ROOT / "legacy" / "meshes"

app = FastAPI(title="GP8 Web GUI")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class GuiNode(Node):
    """ROS 2 node for the web GUI."""

    def __init__(self) -> None:
        super().__init__("gui_server")
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._rgb_frame: np.ndarray | None = None
        self._overlay_frame: np.ndarray | None = None

        cb_group = ReentrantCallbackGroup()

        # Camera subscriptions
        self.create_subscription(
            Image, "/camera/color/image_raw", self._rgb_cb, 10,
            callback_group=cb_group,
        )
        self.create_subscription(
            Image, "/detection_overlay", self._overlay_cb, 10,
            callback_group=cb_group,
        )

        # Device monitor
        self.monitor = DeviceMonitor(self)

        # Manual controller
        self.controller = ManualController(self, cb_group)

        self.get_logger().info("GUI server node initialized")

    def _rgb_cb(self, msg: Image) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            with self._lock:
                self._rgb_frame = frame
        except Exception as e:
            self.get_logger().error(f"RGB conversion error: {e}")

    def _overlay_cb(self, msg: Image) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            with self._lock:
                self._overlay_frame = frame
        except Exception as e:
            self.get_logger().error(f"Overlay conversion error: {e}")

    def get_video_frame(self) -> np.ndarray | None:
        """Get best available frame (overlay > raw > None)."""
        with self._lock:
            if self._overlay_frame is not None:
                return self._overlay_frame.copy()
            if self._rgb_frame is not None:
                return self._rgb_frame.copy()
        return None


# Global node reference (set in main())
_node: GuiNode | None = None


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/robot/urdf")
async def robot_urdf():
    """Serve URDF with mesh paths rewritten to web URLs."""
    urdf_text = URDF_PATH.read_text(encoding="utf-8")
    # Rewrite mesh paths: ../../legacy/meshes/visual/X.stl -> /robot/meshes/visual/X.stl
    urdf_text = re.sub(
        r'filename="[^"]*meshes/(visual|collision)/([^"]+)"',
        r'filename="/robot/meshes/\1/\2"',
        urdf_text,
    )
    return Response(content=urdf_text, media_type="application/xml")


@app.get("/robot/meshes/{mesh_type}/{filename}")
async def robot_mesh(mesh_type: str, filename: str):
    """Serve STL mesh files."""
    mesh_path = MESHES_DIR / mesh_type / filename
    if not mesh_path.exists() or not mesh_path.is_file():
        return Response(status_code=404, content="Mesh not found")
    return FileResponse(mesh_path, media_type="application/octet-stream")


@app.get("/stream/video")
async def video_stream():
    async def generate():
        while True:
            if _node is None:
                await asyncio.sleep(0.1)
                continue
            frame = _node.get_video_frame()
            if frame is None:
                await asyncio.sleep(0.5)
                continue
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg.tobytes()
                + b"\r\n"
            )
            await asyncio.sleep(0.1)  # ~10 FPS

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws/status")
async def ws_status(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            if _node is not None:
                status = _node.monitor.get_status()
                # Also include EE info
                ee_status = _node.controller.get_ee_status()
                status["ee"] = ee_status
                await websocket.send_json(status)
            await asyncio.sleep(0.5)  # 2 Hz
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


@app.websocket("/ws/control")
async def ws_control(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            if _node is None:
                await websocket.send_json({"error": "Node not ready"})
                continue

            action = data.get("action")
            if action == "activate":
                result = _node.controller.set_active(True)
            elif action == "deactivate":
                result = _node.controller.set_active(False)
            elif action == "key":
                key = data.get("key", "")
                # Run blocking handle_key in thread pool
                result = await asyncio.to_thread(_node.controller.handle_key, key)
            else:
                result = {"error": f"Unknown action: {action}"}
            await websocket.send_json(result)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


def main():
    global _node
    import uvicorn

    rclpy.init()
    _node = GuiNode()

    # Spin rclpy in daemon thread
    executor = MultiThreadedExecutor()
    executor.add_node(_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    _node.get_logger().info("Starting web GUI on http://localhost:8080")

    try:
        uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
    finally:
        _node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
