from tetra_model_zoo.googlenet.model import MODEL_NAME, GoogLeNet
from tetra_model_zoo.imagenet_classifier.test_utils import run_imagenet_classifier_test


def test_numerical():
    run_imagenet_classifier_test(GoogLeNet.from_pretrained(), MODEL_NAME)
