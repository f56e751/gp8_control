"""GUI helpers for the recycling env.

Used by the two viewer entry points (`gym_env.py`'s `__main__` and
`verify_throw_chain_gui.py`). Only imported by callers that are launching
the viewer, so it's safe to depend on the MuJoCo / OpenCV runtime context
without forcing those imports on headless paths.
"""

from __future__ import annotations

import time
from typing import Any, Callable


def make_throttled_viewer_hook(
    viewer: Any,
    env: Any,
    *,
    cv2_module: Any | None = None,
    camera_window: str | None = None,
    refresh_hz: float = 60.0,
) -> Callable[[], bool]:
    """Build a step_hook closure for `env.set_sim_step_hook(...)` that:

      * caps `viewer.sync()` to `refresh_hz` per wall-clock second
        (the step_hook fires per sim tick at ~500 Hz; per-tick sync
        accumulates hundreds of ms of wall-clock during long actions
        like push after-trajectories or clamped-throw wait_time);
      * optionally refreshes the OpenCV camera-overlay window on the
        same throttle when `cv2_module` is non-None and `env.use_camera`
        is True;
      * returns False (terminating the sim-tick loop early) when the
        user presses 'q' in the OpenCV window, otherwise mirrors
        `viewer.is_running()`.

    The simulation itself is unchanged — only the visualization update
    rate is capped.
    """
    refresh_period = 1.0 / max(refresh_hz, 1e-3)
    last_sync = [time.perf_counter()]
    quit_requested = [False]

    def step_hook() -> bool:
        now = time.perf_counter()
        if now - last_sync[0] < refresh_period:
            return viewer.is_running() and not quit_requested[0]
        last_sync[0] = now
        viewer.sync()
        if cv2_module is not None and getattr(env, "use_camera", False):
            try:
                frame_bgr, detection, detections = env._capture_detection()
                overlay = env.render_tracking_overlay(
                    frame_bgr=frame_bgr,
                    detection=detection,
                    detections=detections,
                    tracking=env._last_tracking,
                )
                cv2_module.imshow(camera_window, overlay)
                if cv2_module.waitKey(1) & 0xFF == ord("q"):
                    quit_requested[0] = True
                    return False
            except Exception:
                pass
        return viewer.is_running()

    return step_hook
