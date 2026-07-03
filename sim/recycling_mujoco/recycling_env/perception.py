from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    detected: bool
    bbox: tuple[int, int, int, int] | None
    center_px: tuple[float, float] | None
    area: float
    mask: np.ndarray | None


class ColorSegmentationDetector:
    def __init__(
        self,
        lower_hsv: np.ndarray,
        upper_hsv: np.ndarray,
        kernel_size: int = 5,
        min_area: float = 50.0,
    ) -> None:
        self.lower_hsv = np.asarray(lower_hsv, dtype=np.uint8)
        self.upper_hsv = np.asarray(upper_hsv, dtype=np.uint8)
        self.kernel_size = int(kernel_size)
        self.min_area = float(min_area)
        self.kernel = np.ones((self.kernel_size, self.kernel_size), dtype=np.uint8)

    def detect_all(self, bgr_image: np.ndarray) -> list[Detection]:
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        detections: list[Detection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            center_px = (x + w / 2.0, y + h / 2.0)
            detections.append(Detection(True, (x, y, w, h), center_px, area, mask))

        detections.sort(
            key=lambda detection: (
                detection.center_px[0] if detection.center_px is not None else float("inf"),
                -(detection.area),
            )
        )
        return detections

    def detect(self, bgr_image: np.ndarray) -> Detection:
        detections = self.detect_all(bgr_image)
        if not detections:
            hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
            return Detection(False, None, None, 0.0, mask)

        return max(detections, key=lambda detection: detection.area)

    def draw_detection(
        self,
        bgr_image: np.ndarray,
        detection: Detection,
        target_px: tuple[float, float] | None = None,
        bbox_color: tuple[int, int, int] = (255, 0, 0),
        center_color: tuple[int, int, int] = (0, 0, 255),
        target_color: tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        return self.draw_detections(
            bgr_image=bgr_image,
            detections=[detection] if detection.detected else [],
            target_px=target_px,
            primary_index=0 if detection.detected else None,
            bbox_color=bbox_color,
            center_color=center_color,
            target_color=target_color,
        )

    def draw_detections(
        self,
        bgr_image: np.ndarray,
        detections: list[Detection],
        target_px: tuple[float, float] | None = None,
        primary_index: int | None = None,
        bbox_color: tuple[int, int, int] = (255, 0, 0),
        center_color: tuple[int, int, int] = (0, 0, 255),
        target_color: tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        overlay = bgr_image.copy()

        if target_px is not None:
            target_x = int(round(target_px[0]))
            target_y = int(round(target_px[1]))
            cv2.line(overlay, (target_x, 0), (target_x, overlay.shape[0]), target_color, 1)
            cv2.line(overlay, (0, target_y), (overlay.shape[1], target_y), target_color, 1)

        if detections:
            for idx, detection in enumerate(detections):
                if not detection.detected or detection.bbox is None or detection.center_px is None:
                    continue

                x, y, w, h = detection.bbox
                center_x, center_y = detection.center_px
                is_primary = primary_index == idx
                line_color = bbox_color if is_primary else (255, 255, 0)
                dot_color = center_color if is_primary else (0, 165, 255)
                line_width = 2 if is_primary else 1
                cv2.rectangle(overlay, (x, y), (x + w, y + h), line_color, line_width)
                cv2.circle(overlay, (int(round(center_x)), int(round(center_y))), 4, dot_color, -1)
                cv2.putText(
                    overlay,
                    f"id={idx} center=({center_x:.1f}, {center_y:.1f}) area={detection.area:.0f}",
                    (x, max(20, y - 10 - idx * 14)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        else:
            cv2.putText(
                overlay,
                "no detection",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return overlay
