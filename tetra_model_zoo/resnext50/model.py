from __future__ import annotations

import torchvision.models as tv_models

from tetra_model_zoo.imagenet_classifier.model import ImagenetClassifier

MODEL_NAME = "resnext50"


class ResNeXt50(ImagenetClassifier):
    @staticmethod
    def from_pretrained(weights: str = "IMAGENET1K_V2") -> ImagenetClassifier:
        net = tv_models.resnext50_32x4d(weights=weights)
        return ImagenetClassifier(net)
