from __future__ import annotations

import torch
import torch.nn as nn

from tetra_model_zoo.utils.asset_loaders import SourceAsRoot, maybe_download_s3_data
from tetra_model_zoo.utils.input_spec import InputSpec

AOTGAN_SOURCE_REPOSITORY = "https://github.com/researchmm/AOT-GAN-for-Inpainting/"
AOTGAN_SOURCE_REPO_COMMIT = "418034627392289bdfc118d62bc49e6abd3bb185"
MODEL_NAME = "aotgan"
SUPPORTED_PRETRAINED_MODELS = set(["celebahq", "places2"])
WEIGHTS_HELP_MSG = "Following two pre-trained model options are available for AOTGAN: 'celeabhq' or 'places2'."
DEFAULT_WEIGHTS = "celebahq"
MODEL_ASSET_VERSION = "1"


class AOTGAN(torch.nn.Module):
    """Exportable AOTGAN for Image inpainting"""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @staticmethod
    def from_pretrained(ckpt_name: str = DEFAULT_WEIGHTS):
        if ckpt_name not in SUPPORTED_PRETRAINED_MODELS:
            raise ValueError(
                "Unsupported pre_trained model requested. Please provide either 'celeabhq' or 'places2'."
            )
        downloaded_model_path = maybe_download_s3_data(
            f"aotgan/v1/pretrained_models/{ckpt_name}/G0000000.pt",
            MODEL_NAME,
        )
        with SourceAsRoot(
            AOTGAN_SOURCE_REPOSITORY, AOTGAN_SOURCE_REPO_COMMIT, MODEL_NAME
        ):
            from src.model.aotgan import InpaintGenerator

            # AOT-GAN InpaintGenerator uses ArgParser to
            # initialize model and it uses following two parameters
            #  - rates: default value [1, 2, 4, 8]
            #  - block_num: default value 8
            # creating dummy class with default values to set the same
            class InpaintArgs:
                def __init__(self):
                    self.rates = [1, 2, 4, 8]
                    self.block_num = 8

            args = InpaintArgs()
            model = InpaintGenerator(args)
            model.load_state_dict(torch.load(downloaded_model_path, map_location="cpu"))
            return AOTGAN(model)

    def forward(self, image: torch.Tensor, mask: torch.Tensor):
        """
        Run AOTGAN Inpaint Generator on `image` with given `mask`
        and generates new high-resolution in-painted image.

        Parameters:
            image: Pixel values pre-processed of shape [N, C, H, W]
                    Range: float[0, 1]
                    3-channel color Space: BGR
            mask: Pixel values pre-processed to have have mask values either 0. or 1.
                    Range: float[0, 1] and only values of 0. or 1.
                    1-channel binary image.

        Returns:
            In-painted image for given image and mask of shape [N, C, H, W]
            Range: float[0, 1]
            3-channel color space: RGB
        """
        return self.model(image, mask)

    def get_input_spec(
        self,
        image_size: tuple[int, int] | int = 512,
        batch_size: int = 1,
        num_channels: int = 3,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on TetraHub.
        """
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        return {
            "image": ((batch_size, num_channels, *image_size), "float32"),
            "mask": ((batch_size, 1, *image_size), "float32"),
        }
