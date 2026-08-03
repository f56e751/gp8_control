"""Pure helpers for the schema-v2 four-corner perception boxes."""

from __future__ import annotations

import numpy as np


def as_bbox(box) -> np.ndarray:
    """Return a finite ``(4, 3)`` float array or raise ``ValueError``."""
    bbox = np.asarray(box, dtype=float)
    if bbox.shape != (4, 3) or not np.all(np.isfinite(bbox)):
        raise ValueError(f"expected a finite (4, 3) bounding box, got {bbox.shape}")
    return bbox


def bbox_center(box: np.ndarray) -> np.ndarray:
    """Return the centroid of the four projected corners."""
    return np.mean(box, axis=0)


def bbox_to_base(
    box: np.ndarray,
    *,
    ref_x: float,
    ref_y: float,
    ref_z: float,
    scale_x: float,
    scale_y: float,
    y_back_projection: float,
    z_offset: float,
) -> np.ndarray:
    """Transform every belt-frame corner to the corrected robot base plane."""
    return np.column_stack((
        ref_x + scale_x * box[:, 0],
        ref_y + scale_y * box[:, 1] - y_back_projection,
        np.full(4, ref_z + z_offset),
    ))
