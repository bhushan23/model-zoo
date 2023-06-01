from tetra_model_zoo.imagenet_classifier.test_utils import run_imagenet_classifier_test
from tetra_model_zoo.resnet50.model import MODEL_NAME, ResNet50


def test_numerical():
    run_imagenet_classifier_test(
        ResNet50.from_pretrained(), MODEL_NAME, probability_threshold=0.45
    )
