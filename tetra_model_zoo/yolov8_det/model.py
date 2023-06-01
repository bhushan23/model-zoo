from __future__ import annotations

import torch
import torch.nn as nn
from ultralytics import YOLO as ultralytics_YOLO

from tetra_model_zoo.utils.input_spec import InputSpec
from tetra_model_zoo.yolo.utils import (
    get_most_likely_score,
    transform_box_layout_xywh2xyxy,
)

MODEL_ASSET_VERSION = "1"
MODEL_NAME = "yolov8_det"

SUPPORTED_WEIGHTS = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
]
DEFAULT_WEIGHTS = "yolov8n.pt"


class YoloV8Detector(nn.Module):
    """Exportable YoloV8 bounding box detector, end-to-end."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @staticmethod
    def from_pretrained(ckpt_name: str = DEFAULT_WEIGHTS):
        model = ultralytics_YOLO(ckpt_name).model
        model.eval()
        return YoloV8Detector(model)

    def forward(self, image: torch.Tensor):
        """
        Run YoloV8 on `image`, and produce a predicted set of bounding boxes and associated class probabilities.

        Parameters:
            image: Pixel values pre-processed for encoder consumption.
                    Range: float[0, 1]
                    3-channel Color Space: RGB

        Returns:
            boxes: Shape [batch, num preds, 4] where 4 == (center_x, center_y, w, h)
            class scores multiplied by confidence: Shape [batch, num_preds, # of classes (typically 80)]
        """
        predictions, *_ = self.model(image)
        boxes, scores, classes = yolov8_detect_postprocess(predictions)
        return boxes, scores, classes

    def get_input_spec(
        self,
        image_size: tuple[int, int] | int = 640,
        batch_size: int = 1,
        num_channels: int = 3,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on TetraHub.
        """
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        return {"image": ((batch_size, num_channels, *image_size), "float32")}


def yolov8_detect_postprocess(detector_output: torch.Tensor):
    """
    Post processing to break YoloV8 detector output into multiple, consumable tensors (eg. for NMS).
        such as bounding boxes, scores and classes.

    Parameters:
        detector_output: torch.Tensor
            The output of Yolo Detection model
            Shape is [batch, k, num_preds]
                where, k = # of classes + 4
                k is structured as follows [boxes (4) : # of classes]
                and boxes are co-ordinates [x_center, y_center, w, h]

    Returns:
        boxes: torch.Tensor
            Bounding box locations. Shape is [batch, num preds, 4] where 4 == (x1, y1, x2, y2)
        scores: torch.Tensor
            class scores multiplied by confidence: Shape is [batch, num_preds]
        class_idx: torch.tensor
            Shape is [batch, num_preds] where the last dim is the index of the most probable class of the prediction.
    """
    # Break output into parts
    detector_output = torch.permute(detector_output, [0, 2, 1])
    boxes = detector_output[:, :, :4]
    scores = detector_output[:, :, 4:]

    # Convert boxes to (x1, y1, x2, y2)
    boxes = transform_box_layout_xywh2xyxy(boxes)

    # Get class ID of most likely score.
    scores, class_idx = get_most_likely_score(scores)

    return boxes, scores, class_idx
