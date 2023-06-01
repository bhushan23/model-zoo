from tetra_model_zoo.imagenet_classifier.test_utils import run_imagenet_classifier_test
from tetra_model_zoo.mobilenet_v3_small.model import MODEL_NAME, MobileNetV3Small


def test_numerical():
    run_imagenet_classifier_test(MobileNetV3Small.from_pretrained(), MODEL_NAME)
