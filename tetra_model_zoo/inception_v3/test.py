from tetra_model_zoo.imagenet_classifier.test_utils import run_imagenet_classifier_test
from tetra_model_zoo.inception_v3.model import MODEL_NAME, InceptionNetV3


def test_numerical():
    run_imagenet_classifier_test(InceptionNetV3.from_pretrained(), MODEL_NAME)
