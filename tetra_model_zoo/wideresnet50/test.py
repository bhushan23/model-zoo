from tetra_model_zoo.imagenet_classifier.test_utils import run_imagenet_classifier_test
from tetra_model_zoo.wideresnet50.model import MODEL_NAME, WideResNet50


def test_numerical():
    run_imagenet_classifier_test(WideResNet50.from_pretrained(), MODEL_NAME)
