from __future__ import annotations

import torch
import torch.nn as nn

from tetra_model_zoo.utils.asset_loaders import SourceAsRoot, download_data
from tetra_model_zoo.utils.input_spec import InputSpec
from tetra_model_zoo.yolo.utils import detect_postprocess

YOLOV6_SOURCE_REPOSITORY = "https://github.com/meituan/YOLOv6"
YOLOV6_SOURCE_REPO_COMMIT = "55d80c317edd0fb5847e599a1802d394f34a3141"
MODEL_ASSET_VERSION = "1"
MODEL_NAME = "yolov6"

WEIGHTS_PATH = "https://github.com/meituan/YOLOv6/releases/download/0.4.0/"
DEFAULT_WEIGHTS = "yolov6n.pt"


class YoloV6(torch.nn.Module):
    """Exportable YoloV6 bounding box detector, end-to-end."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    # All image input spatial dimensions should be a multiple of this stride.
    STRIDE_MULTIPLE = 32

    @staticmethod
    def from_pretrained(ckpt_name: str = DEFAULT_WEIGHTS):
        model_url = f"{WEIGHTS_PATH}{ckpt_name}"
        model = _load_yolov6_source_model_from_weights(model_url)
        return YoloV6(model)

    def forward(self, image: torch.Tensor):
        """
        Run YoloV6 on `image`, and produce a predicted set of bounding boxes and associated class probabilities.

        Parameters:
            image: Pixel values pre-processed for encoder consumption.
                   Range: float[0, 1]
                   3-channel Color Space: RGB

        Returns:
            boxes: Shape [batch, num preds, 4] where 4 == (center_x, center_y, w, h)
            class scores multiplied by confidence: Shape [batch, num_preds, # of classes (typically 80)]
        """
        predictions = self.model(image)
        return detect_postprocess(predictions)

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


def _load_yolov6_source_model_from_weights(ckpt_path: str) -> torch.nn.Module:
    model_path = download_data(ckpt_path, MODEL_NAME)
    with SourceAsRoot(YOLOV6_SOURCE_REPOSITORY, YOLOV6_SOURCE_REPO_COMMIT, MODEL_NAME):
        from yolov6.layers.common import RepVGGBlock
        from yolov6.utils.checkpoint import load_checkpoint

        model = load_checkpoint(model_path, map_location="cpu", inplace=True, fuse=True)
        model.export = True

        for layer in model.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()
            elif isinstance(layer, nn.Upsample) and not hasattr(
                layer, "recompute_scale_factor"
            ):
                layer.recompute_scale_factor = None  # torch 1.11.0 compatibility
        return model
