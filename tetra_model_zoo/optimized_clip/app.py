from __future__ import annotations

from typing import Tuple

import clip
import torch
from PIL.Image import Image

from tetra_model_zoo.utils.input_spec import InputSpec


class OptimizedClipApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with Clip.

    The app uses 1 model:
        * OptimizedClip

    For a given image input, the app will:
        * pre-process the image
        * pre-process the text
        * Run Clip inference
    """

    def __init__(
        self,
        clip: torch.nn.Module,
    ):
        # Open AI Clip
        self.image_encoder = clip.image_encoder
        self.text_encoder = clip.text_encoder
        # Preprocess Compose function from Open AI clip
        self.preprocess = clip.preprocess

    def predict(self, *args, **kwargs):
        # See predict_similarity.
        return self.predict_similarity(*args, **kwargs)

    def predict_similarity(
        self, image: torch.Tensor, text: torch.Tensor
    ) -> torch.Tensor:
        """
        Inputs:
            image: torch.Tensor (Shape: [1, 3, 224, 224])
                Processed image tensor with values normalized to be between 0-1.
            text: torch.Tensor (Shape: [1, 77])
                Processed text tensor to be tokenized.

        Outputs:
            logits_per_image: torch.Tensor

                Given a batch of images and a batch of text tokens, returns a tensor,
                containing the logit scores corresponding to each image per text input.
                The values are similarity scores between the corresponding image and
                text features, times 100. The logits of text per image can be computed
                by doing a transpose.

        """
        with torch.no_grad():
            image_features = self.image_encoder(image)
            text_features = self.text_encoder(text)
            logits_per_image = image_features @ text_features
        return logits_per_image.cpu().numpy()

    def process_image(self, image: Image) -> torch.Tensor:
        """Process image before calling forward.

        Inputs:
            image: PIL.Image
                Image loaded by Pillow must be provided.
                Example: image = Image.open('<path>')

        Outputs:
            Layout: RGB
            processed_image: torch.Tensor (shape [1, 3, 224, 224])
                The image is converted to torch tensor and normalized
                to be in the range of 0-1.


        """
        return self.preprocess(image).unsqueeze(0)

    def process_text(self, text: str) -> torch.Tensor:
        """Process text into tokens for forward call.

        Input:
            text: str
                Text prompt intended for inference.
                Example: "golden hour"

        Output:
            tokenized_tensor: torch.Tensor (shape: [1, 77])
            Example: tensor([[49406,  3878,  2232, 49407, 0, 0...]])

        """
        return clip.tokenize(text)

    def get_input_spec(
        self,
        image_size: Tuple[int, int] = (224, 224),
        text_size: Tuple[int, int] = (3, 77),
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the tetra_hub python API to declare
        # the model input specification upon submitting a profile job.
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        return {
            "image": ((1, 3, *image_size), "float32"),
            "text": (text_size, "int32"),
        }
