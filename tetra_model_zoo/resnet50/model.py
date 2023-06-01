from __future__ import annotations

import torchvision.models as tv_models

from tetra_model_zoo.imagenet_classifier.model import ImagenetClassifier

MODEL_NAME = "resnet50"


class ResNet50(ImagenetClassifier):
    @staticmethod
    def from_pretrained(weights: str = "IMAGENET1K_V2") -> ImagenetClassifier:
        net = tv_models.resnet50(weights=weights)
        return ImagenetClassifier(net)
