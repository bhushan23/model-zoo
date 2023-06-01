from __future__ import annotations

import torchvision.models as tv_models

from tetra_model_zoo.imagenet_classifier.model import ImagenetClassifier

MODEL_NAME = "mobilenet_v2"


class MobileNetV2(ImagenetClassifier):
    @staticmethod
    def from_pretrained(weights: str = "IMAGENET1K_V2") -> ImagenetClassifier:
        net = tv_models.mobilenet_v2(weights=weights)
        return ImagenetClassifier(net)
