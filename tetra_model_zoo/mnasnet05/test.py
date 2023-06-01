from tetra_model_zoo.imagenet_classifier.test_utils import run_imagenet_classifier_test
from tetra_model_zoo.mnasnet05.model import MODEL_NAME, MNASNet05


def test_numerical():
    run_imagenet_classifier_test(
        MNASNet05.from_pretrained(), MODEL_NAME, probability_threshold=0.69
    )
