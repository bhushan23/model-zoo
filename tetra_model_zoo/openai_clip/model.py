from typing import Tuple

import clip
import torch
import torchvision

from tetra_model_zoo.utils.asset_loaders import callback_with_retry
from tetra_model_zoo.utils.input_spec import InputSpec

PRETRAINED_WEIGHTS = "ViT-B/16"
MODEL_NAME = "clip"
MODEL_ASSET_VERSION = "1"


def load_clip():
    """Downloading pretrained weights via OpenAI and loading them."""
    return clip.load(PRETRAINED_WEIGHTS)


class Clip(torch.nn.Module):
    def __init__(
        self,
        text_encoder: torch.nn.Module,
        image_encoder: torch.nn.Module,
        preprocess: torchvision.transforms.transforms.Compose,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.preprocess = preprocess

    @staticmethod
    def from_pretrained():
        net, preprocess = callback_with_retry(num_retries=5, callback=load_clip)
        return Clip.from_source_model(net, preprocess)

    @staticmethod
    def from_source_model(net, preprocess):
        net = net.eval()
        text_encoder = ClipTextEncoder(net)
        image_encoder = ClipImageEncoder(net)
        return Clip(text_encoder, image_encoder, preprocess)


class ClipTextEncoder(torch.nn.Module):
    def __init__(self, net: torch.nn.Module):
        super().__init__()
        """ Wrapper for OpenAI CLIP."""
        self.net = net
        self.eot_token = 49407

    def forward(self, text: torch.Tensor):
        """Forward call on Open AI CLIP model.

        Inputs:
            text: torch.Tensor (Shape: [1, 77] context_length=77)
                Processed text tensor to be tokenized.

        Outputs:
            text_features: torch.Tensor [512 (transformer_width), num_text_prompts]
                Raw text features are returned. When multiplied to image features,
                you can obtain a matrix of cosine similarities between the
                corresponding image and text input.

        """
        clipped_text = torch.clip(text, min=0, max=self.eot_token)
        text_features = self.net.encode_text(clipped_text)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def get_input_spec(
        self,
        text_size: Tuple[int, int] = (1, 77),
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the tetra_hub python API to declare
        # the model input specification upon submitting a profile job.
        return {
            "text": (text_size, "int32"),
        }


class ClipImageEncoder(torch.nn.Module):
    def __init__(self, net: torch.nn.Module):
        super().__init__()
        """ Wrapper for OpenAI Clip."""
        self.net = net
        self.eot_token = 49407

    def forward(self, image: torch.Tensor):
        """Forward call on Open AI Clip model.

        Inputs:
            image: torch.Tensor (Shape: [1, 3, 224, 224])
                Processed image tensor with values normalized to be between 0-1.
                Channel Layout: RGB

        Outputs:
            image_features: torch.Tensor [num_images, 512 (transformer_width)]
                Raw image features (multiplied to 100) are returned.
                When multiplied to text features, you can obtain a
                matrix of cosine similarities between the corresponding image and
                text input.

        """
        image_features = self.net.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return self.net.logit_scale.exp() * image_features

    def get_input_spec(
        self,
        image_size: Tuple[int, int] = (224, 224),
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the tetra_hub python API to declare
        # the model input specification upon submitting a profile job.
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        return {
            "image": ((1, 3, *image_size), "float32"),
        }
