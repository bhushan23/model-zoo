from tetra_model_zoo.convnext_tiny.model import MODEL_NAME, ConvNextTiny
from tetra_model_zoo.imagenet_classifier.test_utils import run_imagenet_classifier_test


def test_numerical():
    run_imagenet_classifier_test(ConvNextTiny.from_pretrained(), MODEL_NAME)
