from __future__ import annotations

import torchvision.models as tv_models

from tetra_model_zoo.imagenet_classifier.model import ImagenetClassifier

MODEL_NAME = "resnext101"


class ResNeXt101(ImagenetClassifier):
    @staticmethod
    def from_pretrained(weights: str = "IMAGENET1K_V2") -> ImagenetClassifier:
        net = tv_models.resnext101_32x8d(weights=weights)
        return ImagenetClassifier(net)
