from __future__ import annotations

from typing import Callable, List

import torch
from PIL.Image import Image
from torch.nn import functional as F

from tetra_model_zoo.utils.image_processing import (
    app_to_net_image_inputs,
    torch_tensor_to_PIL_image,
)

PRE_PAD = 10
SCALE = 4


class RealESRGANApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with RealESRGAN.

    The app uses 1 model:
        * RealESRGAN

    For a given image input, the app will:
        * pre-process the image (convert to range[0, 1])
        * Run RealESRGAN inference
        * post-process the image
        * display the input and output side-by-side
    """

    def __init__(self, model: Callable[[torch.Tensor], torch.Tensor]):
        self.model = model

    def predict(self, *args, **kwargs):
        # See upscale_image.
        return self.upscale_image(*args, **kwargs)

    def upscale_image(
        self,
        pixel_values_or_image: torch.Tensor | Image | List[Image],
    ) -> List[Image]:
        """
        Upscale provided images

        Parameters:
            pixel_values_or_image
                PIL image(s)
                or
                numpy array (N H W C x uint8) or (H W C x uint8) -- both RGB channel layout
                or
                pyTorch tensor (N C H W x fp32, value range is [0, 1]), RGB channel layout

        Returns:
                images: List[PIL.Image.Image]
                    A list of upscaled images (one for each input image).
        """
        _, NCHW_fp32_torch_frames = app_to_net_image_inputs(pixel_values_or_image)

        # pre-pad with a value of 10
        NCHW_fp32_torch_frames = F.pad(
            NCHW_fp32_torch_frames, (0, PRE_PAD, 0, PRE_PAD), "reflect"
        )

        # Run prediction
        upscaled_images = self.model(NCHW_fp32_torch_frames)
        if len(upscaled_images.shape) == 3:
            upscaled_images = torch.unsqueeze(upscaled_images, 0)

        # Postprocess -- Remove padding
        # preprocessing used a pre_pad value of 10
        # These weights use a scale of 4
        _, _, h, w = upscaled_images.shape
        upscaled_images = upscaled_images[
            :, :, 0 : h - PRE_PAD * SCALE, 0 : w - PRE_PAD * SCALE
        ]

        return [torch_tensor_to_PIL_image(img) for img in upscaled_images]
