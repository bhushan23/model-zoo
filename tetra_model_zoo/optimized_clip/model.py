# import ane_transformers
from typing import Tuple

import torch
import torchvision
from PIL.Image import Image

from tetra_model_zoo.openai_clip.model import load_clip
from tetra_model_zoo.optimized_clip import transformer_utils
from tetra_model_zoo.utils.asset_loaders import callback_with_retry
from tetra_model_zoo.utils.input_spec import InputSpec

MODEL_NAME = "optimized_clip"


class OptimizedClip(torch.nn.Module):
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
        return OptimizedClip.from_source_model(net, preprocess)

    @staticmethod
    def from_source_model(net, preprocess):
        net = net.eval()
        text_encoder = OptimizedClipTextEncoder(net)
        image_encoder = OptimizedClipImageEncoder(net)
        return OptimizedClip(text_encoder, image_encoder, preprocess)


class OptimizedClipTextEncoder(torch.nn.Module):
    def __init__(self, net: torch.nn.Module):
        super().__init__()
        setattr(
            self,
            "ln_final",
            transformer_utils.LayerNormANE(
                net.transformer.width, eps=transformer_utils.EPS
            ),
        )

        self.text_projection = net.text_projection
        self.token_embedding = net.token_embedding

        self.positional_embedding = net.positional_embedding
        self.context_length = net.context_length
        transformer = net.transformer
        width = net.transformer.width
        layers = net.transformer.layers

        # Create ane ANE transformer for encoding text
        ane_transformer = transformer_utils.Transformer(
            width, layers, width // 64, self.build_attention_mask()
        )
        ane_transformer.load_state_dict(transformer.state_dict())
        setattr(self, "transformer", ane_transformer)
        self.eot_token = 49407

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask.permute(1, 0).unsqueeze(0).unsqueeze(-2)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """
        Encode the text using the ANE Transformer based modules.

        Inputs:
            text: torch.Tensor (Shape: [1, 77])
                Text tokens.

        Outputs:
            text: torch.Tensor (Shape: [1, 512 (transformer_width)])
                Given a batch of text tokens, returns the text features
                encoded by the language portion of the CLIP model.
        """
        text = torch.clip(text, min=0, max=self.eot_token)
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x.permute(2, 0, 1).unsqueeze(0)
        x = x + self.positional_embedding.transpose(1, 0).unsqueeze(0).unsqueeze(-2)
        x = self.transformer(x)
        x = self.ln_final(x)
        x = x.squeeze(0).permute(1, 2, 0)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, text: str) -> torch.Tensor:
        """Forward call on Open AI Clip model.

        Inputs:
            text: torch.Tensor (Shape: [1, 77] context_length=77)
                Processed text tensor to be tokenized.

        Outputs:
            text_features: torch.Tensor [512 (transformer_width), num_text_prompts]
                Raw text features are returned. When multiplied to image features,
                you can obtain a matrix of cosine similarities between the
                corresponding image and text input.

        """
        text_features = self.encode_text(text)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        text_features = text_features.permute(1, 0)
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


class OptimizedClipImageEncoder(torch.nn.Module):
    def __init__(self, net: torch.nn.Module):
        super().__init__()
        setattr(
            self,
            "vision",
            transformer_utils.VisionTransformer.from_clip(net),
        )
        self.logit_scale = net.logit_scale

    def forward(self, image: Image) -> torch.Tensor:
        """Forward call on Open AI Clip model.

        Inputs:
            image: torch.Tensor (Shape: [num_images, 3, 224, 224])
                Processed image tensor with values normalized to be between 0-1.
                Channel Layout: RGB

        Outputs:
            image_features: torch.Tensor [num_images, 512 (transformer_width)]
                Raw image features (multiplied to 100) are returned.
                When multiplied to text features, you can obtain a
                matrix of cosine similarities between the corresponding image and
                text input.

        """

        image_features = self.vision(image)
        image_features = image_features.squeeze(-1).squeeze(-1)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return self.logit_scale.exp() * image_features

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
