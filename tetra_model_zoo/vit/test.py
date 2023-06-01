from tetra_model_zoo.imagenet_classifier.test_utils import run_imagenet_classifier_test
from tetra_model_zoo.vit.model import MODEL_NAME, VIT


def test_numerical():
    run_imagenet_classifier_test(VIT.from_pretrained(), MODEL_NAME)
