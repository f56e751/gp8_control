import unittest

import numpy as np

from perception.bbox_geometry import as_bbox, bbox_center, bbox_to_base


class TestBoundingBoxGeometry(unittest.TestCase):
    def setUp(self):
        self.box = as_bbox([
            [-0.1, -0.2, 0.0], [0.1, -0.2, 0.0],
            [0.1, 0.2, 0.0], [-0.1, 0.2, 0.0],
        ])

    def test_center_uses_all_corners(self):
        np.testing.assert_allclose(bbox_center(self.box), [0.0, 0.0, 0.0])

    def test_transform_preserves_all_corners_and_applies_delay(self):
        actual = bbox_to_base(
            self.box,
            ref_x=0.45, ref_y=-0.10, ref_z=0.06,
            scale_x=1.0, scale_y=-1.0,
            y_back_projection=0.02, z_offset=0.07,
        )
        expected = np.array([
            [0.35, 0.08, 0.13], [0.55, 0.08, 0.13],
            [0.55, -0.32, 0.13], [0.35, -0.32, 0.13],
        ])
        np.testing.assert_allclose(actual, expected)

    def test_rejects_a_center_point(self):
        with self.assertRaises(ValueError):
            as_bbox([0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
