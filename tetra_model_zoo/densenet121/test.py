from tetra_model_zoo.densenet121.model import MODEL_NAME, DenseNet
from tetra_model_zoo.imagenet_classifier.test_utils import run_imagenet_classifier_test


def test_numerical():
    run_imagenet_classifier_test(DenseNet.from_pretrained(), MODEL_NAME)
