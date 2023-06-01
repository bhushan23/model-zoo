from __future__ import annotations

import torch

from tetra_model_zoo.yolo.app import YoloObjectDetectionApp


class YoloV8DetectionApp(YoloObjectDetectionApp):
    def check_image_size(self, pixel_values: torch.Tensor) -> None:
        """
        YoloV8 does not check for spatial dim shapes for input image
        """
        pass
