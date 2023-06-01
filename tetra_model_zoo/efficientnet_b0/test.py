from tetra_model_zoo.efficientnet_b0.model import MODEL_NAME, EfficientNetB0
from tetra_model_zoo.imagenet_classifier.test_utils import run_imagenet_classifier_test


def test_numerical():
    run_imagenet_classifier_test(EfficientNetB0.from_pretrained(), MODEL_NAME)
