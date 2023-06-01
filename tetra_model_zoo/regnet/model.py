from __future__ import annotations

import torchvision.models as tv_models

from tetra_model_zoo.imagenet_classifier.model import ImagenetClassifier

MODEL_NAME = "regnet"


class RegNet(ImagenetClassifier):
    @staticmethod
    def from_pretrained(weights: str = "IMAGENET1K_V2") -> ImagenetClassifier:
        net = tv_models.regnet_y_400mf(weights=weights)
        return ImagenetClassifier(net)
