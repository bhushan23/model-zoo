import numpy as np
import torch

from tetra_model_zoo.imagenet_classifier.app import ImagenetClassifierApp
from tetra_model_zoo.imagenet_classifier.model import ImagenetClassifier
from tetra_model_zoo.utils.asset_loaders import (
    MODEL_ZOO_ASSET_PATH,
    load_image,
    maybe_download_s3_data,
)

MODEL_ASSET_VERSION = "1"
TEST_IMAGENET_IMAGE = (
    f"{MODEL_ZOO_ASSET_PATH}/imagenet_classifier/v{MODEL_ASSET_VERSION}/dog.jpg"
)

# Class "Samoyed" from https://gist.github.com/ageitgey/4e1342c10a71981d0b491e1b8227328b
TEST_IMAGENET_CLASS = 258


def run_imagenet_classifier_test(
    model: ImagenetClassifier,
    model_name: str,
    asset_version: str = "2",
    probability_threshold: float = 0.7,
) -> None:
    """
    Evaluates the classifier on a test image and validates the output.

    Parameters:
        model: The model to evaluate.
        model_name: Identifier used to lookup the expected output file.
        asset_version: Version of the expected output file to lookup.
        probability_threshold: If the predicited probability for the correct class
            is below this threshold, the method throws an error.
    """

    img = load_image(TEST_IMAGENET_IMAGE, "imagenet_classifier")
    app = ImagenetClassifierApp(model)
    probabilities = app.predict(img)

    data_path = f"{model_name}/v{asset_version}/expected_out.npy"
    expected_out_path = maybe_download_s3_data(data_path, model_name)
    expected_out = np.load(expected_out_path)
    np.testing.assert_allclose(probabilities, expected_out, atol=1e-4)

    predicted_class = torch.argmax(probabilities, dim=0)
    predicted_probability = probabilities[TEST_IMAGENET_CLASS].item()
    assert (
        predicted_probability > probability_threshold
    ), f"Predicted probability {predicted_probability:.3f} is below the threshold {probability_threshold}."
    assert (
        predicted_class == TEST_IMAGENET_CLASS
    ), f"Model predicted class {predicted_class} when correct class was {TEST_IMAGENET_CLASS}."
