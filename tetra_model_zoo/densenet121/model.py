from __future__ import annotations

import torchvision.models as tv_models

from tetra_model_zoo.imagenet_classifier.model import ImagenetClassifier

MODEL_NAME = "densenet121"


class DenseNet(ImagenetClassifier):
    @staticmethod
    def from_pretrained(weights: str = "IMAGENET1K_V1") -> ImagenetClassifier:
        net = tv_models.densenet121(weights=weights)
        return ImagenetClassifier(net)
