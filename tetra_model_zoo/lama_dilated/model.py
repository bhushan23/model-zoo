from __future__ import annotations

import json
import os

import torch
from omegaconf import OmegaConf

from tetra_model_zoo.utils.asset_loaders import MODEL_ZOO_ASSET_PATH, SourceAsRoot
from tetra_model_zoo.utils.input_spec import InputSpec

LAMA_SOURCE_REPOSITORY = "https://github.com/advimman/lama"
LAMA_SOURCE_REPO_COMMIT = "7dee0e4a3cf5f73f86a820674bf471454f52b74f"
MODEL_NAME = "lama_dilated"
MODEL_ASSET_VERSION = "1"
DEFAULT_WEIGHTS = "lama-dilated_celeba-hq"


class LamaDilated(torch.nn.Module):
    """Exportable LamaDilated inpainting algorithm by Samsung Research."""

    def __init__(
        self,
        lama_dilated_model: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.model = lama_dilated_model

    @staticmethod
    def from_pretrained(weights_name: str = DEFAULT_WEIGHTS) -> LamaDilated:
        """Load LamaDilated from a weights file created by the source LaMa repository."""

        # Load PyTorch model from disk
        lama_dilated_model = _load_lama_dilated_source_model_from_weights(weights_name)

        return LamaDilated(lama_dilated_model)

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Run LamaDilated on `image` and `mask`, and produce an image with mask area inpainted.

        Parameters:
            image: Pixel values pre-processed for encoder consumption.
                   Range: float[0, 1]
                   3-channel Color Space: RGB

            mask: Pixel values pre-processed to have have mask values either 0. or 1.
                  Range: float[0, 1] and only values of 0. or 1.
                  1-channel binary image.

        Returns:
            inpainted_image: Pixel values
                   Range: float[0, 1]
                   3-channel Color Space: RGB
        """

        masked_img = image * (1 - mask)

        if self.model.concat_mask:
            masked_img = torch.cat([masked_img, mask], dim=1)

        predicted_image = self.model.generator(masked_img)
        inpainted = mask * predicted_image + (1 - mask) * image
        return inpainted

    def get_input_spec(
        self,
        image_size: tuple[int, int] | int,
        batch_size: int = 1,
        num_channels: int = 3,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the tetra_hub python API to declare
        # the model input specification upon submitting a profile job.
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        return {
            "image": ((batch_size, num_channels, *image_size), "float32"),
            "mask": ((batch_size, 1, *image_size), "float32"),
        }


def _get_weightsfile_from_name(weights_name: str):
    """Convert from names of weights files to the url for the weights file"""
    return f"{MODEL_ZOO_ASSET_PATH}/lama_dilated/v{MODEL_ASSET_VERSION}/checkpoints/{weights_name}.ckpt"


def _get_config_url():
    """Get the url for the config file"""
    return f"{MODEL_ZOO_ASSET_PATH}/lama_dilated/v{MODEL_ASSET_VERSION}/checkpoints/training_config.json"


def _load_lama_dilated_source_model_from_weights(weights_name: str) -> torch.nn.Module:
    # Load LamaDilated model from the source repository using the given weights.
    weights_url = _get_weightsfile_from_name(weights_name)
    config_url = _get_config_url()

    with SourceAsRoot(LAMA_SOURCE_REPOSITORY, LAMA_SOURCE_REPO_COMMIT, MODEL_NAME):

        # download the weights and config file
        import requests

        response = requests.get(weights_url)
        weights_path = os.path.join(os.getcwd(), weights_name + ".ckpt")
        with open(weights_path, "wb") as file:
            file.write(response.content)
        print(f"Weights file downloaded as {weights_path}")

        response = requests.get(config_url)
        config_path = os.path.join(os.getcwd(), "training_config.json")
        with open(config_path, "wb") as file:
            file.write(response.content)
        print(f"Weights file downloaded as {config_path}")

        # Import module
        from saicinpainting.training.trainers.default import (
            DefaultInpaintingTrainingModule,
        )

        # Pass config as needed to create the module for tracing.
        with open(config_path, "r") as file:
            config = json.load(file)
        config = OmegaConf.create(config)
        kwargs = dict(config.training_model)
        kwargs.pop("kind")
        kwargs["use_ddp"] = True
        state = torch.load(weights_path, map_location="cpu")
        lama_dilated_model = DefaultInpaintingTrainingModule(config, **kwargs)
        lama_dilated_model.load_state_dict(state["state_dict"], strict=False)
        lama_dilated_model.on_load_checkpoint(state)
        lama_dilated_model.freeze()
        return lama_dilated_model
