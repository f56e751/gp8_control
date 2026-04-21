"""Throw-trajectory FCN predictor (in-process, no ZMQ).

The ROS 1 port originally fronted this model with a ZMQ REP server because
``motoman_driver`` ran under Python 2 while PyTorch only supported Python 3.
On ROS 2 Humble everything is Python 3.10 so the model can live inside the
same process as the controller — no socket, no extra terminal, no
serialisation overhead per throw.

Input  : grasp_xy (2), target_xy (2), target_distance (1)  → 5 floats
Output : 50 trajectory weights (10 basis × 5 joints) +
         log(throw_time) + logit(release_ratio_eta)         → 52 floats
"""

from __future__ import annotations

import os

import numpy as np
import torch

from gp8_control.model.fcn import FCN


_DEFAULT_WEIGHT = "NN_newprimitive2_40000datapoint_500epochs_v2_B10.pt"


def _default_weight_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    # trajectory/ sits next to model/
    return os.path.abspath(os.path.join(here, os.pardir, "model", _DEFAULT_WEIGHT))


class TrajectoryPredictor:
    """Wraps the FCN checkpoint in a ``predict(...)`` call."""

    FCN_SHAPE = [5, 256, 512, 256, 52]

    def __init__(self, weight_path: str | None = None, device: str = "cpu") -> None:
        path = weight_path or os.environ.get("GP8_TRAJECTORY_MODEL") or _default_weight_path()
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Trajectory NN weight not found at {path!r}. "
                "Set GP8_TRAJECTORY_MODEL or copy the .pt into gp8_control/model/."
            )

        self.device = torch.device(device)
        self.model = FCN(self.FCN_SHAPE).to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        self._weight_path = path

    @property
    def weight_path(self) -> str:
        return self._weight_path

    def predict(
        self,
        grasp_xy: tuple[float, float],
        target_xy: tuple[float, float],
        target_distance: float,
    ) -> np.ndarray:
        """Returns 52-float np.ndarray: [w(50), log_T, logit_eta]."""
        x = torch.tensor(
            [grasp_xy[0], grasp_xy[1], target_xy[0], target_xy[1], target_distance],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            y = self.model(x).cpu().numpy()
        return np.asarray(y, dtype=float)
