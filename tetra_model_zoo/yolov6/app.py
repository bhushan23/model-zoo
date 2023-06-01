from __future__ import annotations

import torch

from tetra_model_zoo.yolo.app import YoloObjectDetectionApp


class YoloV6DetectionApp(YoloObjectDetectionApp):
    def check_image_size(self, pixel_values: torch.Tensor) -> None:
        """
        Verify image size is valid model input.
        """
        if len(pixel_values.shape) != 4:
            raise ValueError("Pixel Values must be rank 4: [batch, channels, x, y]")
        if (
            pixel_values.shape[2] % self.model.STRIDE_MULTIPLE != 0
            or pixel_values.shape[3] % self.model.STRIDE_MULTIPLE != 0
        ):
            raise ValueError(
                f"Pixel values must have spatial dimensions (H & W) that are multiples of {self.model.STRIDE_MULTIPLE}."
            )
