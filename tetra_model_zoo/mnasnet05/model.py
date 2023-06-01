from __future__ import annotations

import torchvision.models as tv_models

from tetra_model_zoo.imagenet_classifier.model import ImagenetClassifier

MODEL_NAME = "mnasnet05"


class MNASNet05(ImagenetClassifier):
    @staticmethod
    def from_pretrained(weights: str = "IMAGENET1K_V1") -> ImagenetClassifier:
        net = tv_models.mnasnet0_5(weights=weights)
        return ImagenetClassifier(net)
