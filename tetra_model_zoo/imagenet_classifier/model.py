from typing import Any

import torch

from tetra_model_zoo.utils.input_spec import InputSpec

MODEL_ASSET_VERSION = 1
MODEL_NAME = "imagenet_classifier"
IMAGENET_DIM = 224


class DocstringInheritorMeta(type):
    """
    Ensures that all subclasses of ImagenetClassifier retain the `forward`
    function's docstring.
    """

    def __new__(cls, name, bases, dct):
        new_class = super().__new__(cls, name, bases, dct)
        if hasattr(new_class, "forward"):
            parent_method = getattr(bases[0], "forward", None)
            if parent_method:
                new_class.forward.__doc__ = parent_method.__doc__  # type: ignore
        return new_class


class ImagenetClassifier(torch.nn.Module, metaclass=DocstringInheritorMeta):
    """
    Base class for all Imagenet Classifier models within the model zoo.
    """

    def __init__(self, net: torch.nn.Module):
        """
        Basic initializer which takes in a pretrained classifier network.
        Subclasses can choose to implement their own __init__ and forward methods.
        """
        super().__init__()
        self.net = net

    def forward(self, image_tensor: torch.Tensor):
        """
        Predict class probabilities for an input `image`.

        Parameters:
            image: A [1, 3, 224, 224] image.
                   Assumes image has been resized and normalized using the
                   standard preprocessing method for PyTorch Imagenet models.

                   Pixel values pre-processed for encoder consumption.
                   Range: float[0, 1]
                   3-channel Color Space: RGB

        Returns:
            A [1, 1000] where each value is the log-likelihood of
            the image belonging to the corresponding Imagenet class.
        """
        return self.net(image_tensor)

    def get_input_spec(
        self,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on TetraHub.
        """
        return {"image": ((1, 3, IMAGENET_DIM, IMAGENET_DIM), "float32")}


def trace_imagenet_classifier(model: ImagenetClassifier) -> Any:
    model.eval()
    input_shape = model.get_input_spec()["image"][0]
    return torch.jit.trace(model, [torch.ones(input_shape)])
