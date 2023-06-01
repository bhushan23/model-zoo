from __future__ import annotations

from typing import Callable, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from tetra_model_zoo.ddrnetslim.model import NUM_CLASSES
from tetra_model_zoo.utils.image_processing import (
    app_to_net_image_inputs,
    normalize_image_tranform,
)


def create_color_map(num_classes):
    """
    Assign a random color to each class in the dataset to produce the segmentation mask.

    Inputs:
        num_classes: Number of colors to produce.

    Returns:
        A list of `num_classes` colors in RGB format.
    """
    np.random.seed(42)  # For reproducible results
    color_map = np.random.randint(0, 256, size=(num_classes, 3), dtype=np.uint8)
    color_map[0] = [0, 0, 0]  # Background class, usually black
    return color_map


COLOR_MAP = create_color_map(NUM_CLASSES)


class DDRNetApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with DDRNet.

    The app uses 1 model:
        * DDRNet

    For a given image input, the app will:
        * pre-process the image (convert to range[0, 1])
        * Run DDRNet inference
        * Convert the output segmentation mask into a visual representation
        * Overlay the segmentation mask onto the image and return it
    """

    def __init__(self, model: Callable[[torch.Tensor], torch.Tensor]):
        self.model = model

    def predict(self, *args, **kwargs):
        # See segment_image.
        return self.segment_image(*args, **kwargs)

    def segment_image(
        self,
        pixel_values_or_image: torch.Tensor
        | np.ndarray
        | Image.Image
        | List[Image.Image],
        raw_output: bool = False,
    ) -> List[Image.Image] | np.ndarray:
        """
        Return the input image with the segmentation mask overlayed on it.

        Parameters:
            pixel_values_or_image
                PIL image(s)
                or
                numpy array (N H W C x uint8) or (H W C x uint8) -- both RGB channel layout
                or
                pyTorch tensor (N C H W x fp32, value range is [0, 1]), RGB channel layout

            raw_output: bool
                See "returns" doc section for details.

        Returns:
            If raw_output is true, returns:
                masks: np.ndarray
                    A list of predicted masks.

            Otherwise, returns:
                segmented_images: List[PIL.Image]
                    Images with segmentation map overlaid with an alpha of 0.5.
        """
        NHWC_int_numpy_frames, NCHW_fp32_torch_frames = app_to_net_image_inputs(
            pixel_values_or_image
        )
        input_transform = normalize_image_tranform()
        NCHW_fp32_torch_frames = input_transform(NCHW_fp32_torch_frames)

        with torch.no_grad():
            # pred_mask is 8x downsampled
            pred_masks = self.model(NCHW_fp32_torch_frames)

        # Upsample pred mask to original image size
        # Need to upsample in the probability space, not in class labels
        pred_masks = F.interpolate(
            input=pred_masks,
            size=NCHW_fp32_torch_frames.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        if raw_output:
            return pred_masks.detach().numpy()

        # Create color map and convert segmentation mask to RGB image
        pred_mask_img = torch.argmax(pred_masks, 1)

        # Overlay the segmentation mask on the image. alpha=1 is mask only,
        # alpha=0 is image only.
        out = []
        for i, img_tensor in enumerate(NHWC_int_numpy_frames):
            out.append(
                Image.blend(
                    Image.fromarray(img_tensor),
                    Image.fromarray(COLOR_MAP[pred_mask_img[i]]),
                    alpha=0.5,
                )
            )
        return out
