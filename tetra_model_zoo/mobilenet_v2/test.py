from tetra_model_zoo.imagenet_classifier.test_utils import run_imagenet_classifier_test
from tetra_model_zoo.mobilenet_v2.model import MODEL_NAME, MobileNetV2


def test_numerical():
    run_imagenet_classifier_test(
        MobileNetV2.from_pretrained(), MODEL_NAME, probability_threshold=0.39
    )
