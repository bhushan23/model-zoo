from __future__ import annotations

from typing import List

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL.Image import Image, fromarray


class ESRGANApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with RealESRGAN.

    The app uses 1 model:
        * ESRGAN

    For a given image input, the app will:
        * pre-process the image (convert to range[0, 1])
        * Run ESRGAN inference
        * post-process the image
        * display the input and output side-by-side
    """

    def __init__(self, esrgan_model):
        self.model = esrgan_model

    def predict(self, *args, **kwargs):
        # See upscale_image.
        return self.upscale_image(*args, **kwargs)

    def upscale_image(
        self,
        pixel_values_or_image: torch.Tensor | Image | List[Image],
    ) -> Image:
        """
        Upscale provided images

        Parameters:
            pixel_values_or_image: torch.Tensor
                Input PIL image (before pre-processing) or pyTorch tensor (after image pre-processing).

        Returns:
                images: List[PIL.Image.Image]
                    A list of upscaled images (one for each input image).
        """

        # preprocess
        pixel_values = preprocess_image(pixel_values_or_image)

        # Run prediction
        upscaled_image = self.model(pixel_values)

        # post-process
        output_image = postprocess_image(upscaled_image)

        return output_image


def preprocess_image(image: Image) -> torch.Tensor:
    """
    Convert a raw image to RGB and then into a normalised pyTorch tensor
    that can be used as input to RealESRGAN inference.
    """
    transform = transforms.Compose([transforms.PILToTensor()])  # bgr image
    img: torch.Tensor = transform(image)  # type: ignore
    img = img.float() / 255.0  # int 0 - 255 to float 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img


def postprocess_image(image: Image) -> Image:
    """
    Convert from range[0, 1] to int8 values for display.
    """
    output_img = np.squeeze(image)
    output_img = output_img.detach().numpy().astype(float)
    output_img = np.transpose(output_img, (1, 2, 0))
    output_img = np.clip(output_img * 255.0, 0, 255)
    output_img = output_img.round().astype(np.uint8)
    output = fromarray(output_img)

    return output
