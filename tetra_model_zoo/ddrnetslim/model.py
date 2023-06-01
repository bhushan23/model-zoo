from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from tetra_model_zoo.utils.asset_loaders import SourceAsRoot, download_google_drive
from tetra_model_zoo.utils.input_spec import InputSpec

DDRNET_SOURCE_REPOSITORY = "https://github.com/chenjun2hao/DDRNet.pytorch"
DDRNET_SOURCE_REPO_COMMIT = "bc0e193e87ead839dbc715c48e6bfb059cf21b27"
MODEL_NAME = "ddrnet"
DEFAULT_WEIGHTS_FILE_ID = "1d_K3Af5fKHYwxSo8HkxpnhiekhwovmiP"
DEFAULT_WEIGHTS = "DDRNet23s_imagenet.pth"
MODEL_ASSET_VERSION = "1"
NUM_CLASSES = 19


class DDRNet(torch.nn.Module):
    """Exportable DDRNet image segmenter, end-to-end."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @staticmethod
    def from_pretrained(checkpoint_path: str | None = None):
        """Load DDRNetSlim from a weightfile created by the source DDRNetSlim repository."""
        with SourceAsRoot(
            DDRNET_SOURCE_REPOSITORY, DDRNET_SOURCE_REPO_COMMIT, MODEL_NAME
        ):
            bad_init_file = Path("lib/models/__init__.py")
            if bad_init_file.exists():
                bad_init_file.unlink()

            from lib.models.ddrnet_23_slim import BasicBlock, DualResNet  # type: ignore

            ddrnetslim_model = DualResNet(
                BasicBlock,
                [2, 2, 2, 2],
                num_classes=NUM_CLASSES,
                planes=32,
                spp_planes=128,
                head_planes=64,
                # No need to use aux loss for inference
                augment=False,
            )

            if not checkpoint_path:
                checkpoint_path = download_google_drive(
                    DEFAULT_WEIGHTS_FILE_ID, MODEL_NAME, DEFAULT_WEIGHTS
                )

            pretrained_dict = torch.load(
                checkpoint_path, map_location=torch.device("cpu")
            )
            if "state_dict" in pretrained_dict:
                pretrained_dict = pretrained_dict["state_dict"]
            model_dict = ddrnetslim_model.state_dict()
            pretrained_dict = {
                k[6:]: v
                for k, v in pretrained_dict.items()
                if k[6:] in model_dict.keys()
            }
            model_dict.update(pretrained_dict)
            ddrnetslim_model.load_state_dict(model_dict)

            ddrnetslim_model.to(torch.device("cpu")).eval()

            return DDRNet(ddrnetslim_model)

    def forward(self, image: torch.Tensor):
        """
        Run DDRNetSlim on `image`, and produce a predicted segmented image mask.

        Parameters:
            image: Pixel values pre-processed for encoder consumption.
                   Range: float[0, 1]
                   3-channel Color Space: BGR

        Returns:
            segmented mask per class: Shape [batch, classes, 128, 256]
        """
        return self.model(image)

    def get_input_spec(
        self,
        image_size: tuple[int, int] | int = 320,
        batch_size: int = 1,
        num_channels: int = 3,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on TetraHub. Default resolution is 2048x1024
        so this expects an image where width is twice the height.
        """
        if isinstance(image_size, int):
            image_size = (2 * image_size, image_size)
        return {"image": ((batch_size, num_channels, *image_size), "float32")}
