from __future__ import annotations

import os

import torch

from tetra_model_zoo.utils.asset_loaders import SourceAsRoot
from tetra_model_zoo.utils.input_spec import InputSpec

REALESRGAN_SOURCE_REPOSITORY = "https://github.com/xinntao/Real-ESRGAN"
REALESRGAN_SOURCE_REPO_COMMIT = "5ca1078535923d485892caee7d7804380bfc87fd"
MODEL_NAME = "real-esrgan"
MODEL_ASSET_VERSION = "1"
DEFAULT_WEIGHTS = "realesr-general-x4v3"


class RealESRGAN(torch.nn.Module):
    """Exportable RealESRGAN upscaler, end-to-end."""

    def __init__(
        self,
        realesrgan_model: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.model = realesrgan_model

    @staticmethod
    def from_pretrained(
        weight_path: str = DEFAULT_WEIGHTS,
    ) -> RealESRGAN:
        """Load RealESRGAN from a weightfile created by the source RealESRGAN repository."""

        # Load PyTorch model from disk
        realesrgan_model = _load_realesrgan_source_model_from_weights(weight_path)

        return RealESRGAN(realesrgan_model)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run RealESRGAN on `image`, and produce an upscaled image

        Parameters:
            image: Pixel values pre-processed for GAN consumption.
                   Range: float[0, 1]
                   3-channel Color Space: RGB

        Returns:
            image: Pixel values
                   Range: float[0, 1]
                   3-channel Color Space: RGB
        """
        return self.model(image)

    def get_input_spec(
        self,
        image_size: tuple[int, int] | int = 320,
        batch_size: int = 1,
        num_channels: int = 3,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the tetra_hub python API to declare
        # the model input specification upon submitting a profile job.
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        return {"image": ((batch_size, num_channels, *image_size), "float32")}


def _get_weightsfile_from_name(weights_name: str = "realesr-general-x4v3"):
    """Convert from names of weights files to the url for the weights file"""
    if weights_name == "realesr-general-x4v3":
        return "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
    return ""


def _load_realesrgan_source_model_from_weights(
    weights_name_or_path: str,
) -> torch.nn.Module:
    with SourceAsRoot(
        REALESRGAN_SOURCE_REPOSITORY, REALESRGAN_SOURCE_REPO_COMMIT, MODEL_NAME
    ):
        # Patch path for this load only, since the model source
        # code references modules via a global scope.
        # CWD should be the repository path now
        realesrgan_repo_path = os.getcwd()
        # The official repo omits this folder, which causes import issues
        version_dir = os.path.join(realesrgan_repo_path, "realesrgan/version")
        if not os.path.exists(version_dir):
            os.makedirs(version_dir)

        if os.path.exists(os.path.expanduser(weights_name_or_path)):
            weights_path = os.path.expanduser(weights_name_or_path)
        else:
            weights_path = os.path.join(os.getcwd(), weights_name_or_path + ".pth")
            if not os.path.exists(weights_path):
                # Load RealESRGAN model from the source repository using the given weights.
                # Returns <source repository>.realesrgan.archs.srvgg_arch
                weights_url = _get_weightsfile_from_name(weights_name_or_path)

                # download the weights file
                import requests

                response = requests.get(weights_url)
                with open(weights_path, "wb") as file:
                    file.write(response.content)
                print(f"Weights file downloaded as {weights_path}")

        # necessary import. `archs` comes from the realesrgan repo.
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact

        realesrgan_model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type="prelu",
        )
        pretrained_dict = torch.load(weights_path, map_location=torch.device("cpu"))

        if "params_ema" in pretrained_dict:
            keyname = "params_ema"
        else:
            keyname = "params"
        realesrgan_model.load_state_dict(pretrained_dict[keyname], strict=True)

        return realesrgan_model