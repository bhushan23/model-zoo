from tetra_model_zoo.imagenet_classifier.test_utils import run_imagenet_classifier_test
from tetra_model_zoo.resnext50.model import MODEL_NAME, ResNeXt50


def test_numerical():
    run_imagenet_classifier_test(ResNeXt50.from_pretrained(), MODEL_NAME)
