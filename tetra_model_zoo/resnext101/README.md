[![Tetra AI](https://tetra.ai/img/logo.svg)](https://tetra.ai/)

# [ResNeXt101: Imagenet classifier and general purpose backbone optimized for mobile and edge](https://tetraai.com/model-zoo/resnext101)

ResNeXt101 is a machine learning model that can classify images from the Imagenet dataset.
It can also be used as a backbone in building more complex models for specific use cases.
We present an optimized implementation of ResNext101 suitable to export and run it on-device.

This is based on [ResNeXt101](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py). You can optionally
fine-tune the pre-trained model before walking through the examples below.

## Example & Usage

1. Install the package via pip:
```
pip install tetra_model_zoo[resnext101]
```

2. Load the model & app
```
from tetra_model_zoo.resnext101 import Model
from tetra_model_zoo.resnext101 import App

# check app.py for more details
app = App(Model.from_pretrained())
```

3. Run prediction
```
from tetra_model_zoo.utils.asset_loaders import load_image
from tetra_model_zoo.imagenet_classifier.test_utils import TEST_IMAGENET_IMAGE
from tetra_model_zoo.resnext101 import MODEL_NAME

image = load_image(TEST_IMAGENET_IMAGE, MODEL_NAME)
app.predict(image)
```

See [app.py](../video_classifier/app.py#L137) for more information about e2e usage of the model.

Please refer to our [general instructions on using models](../../#tetra-model-zoo)

### Optimize, Profile, and Validate ResNext101 for a device with Tetra Hub
Using Tetra Hub, you can easily optimize, profile, and validate ResNeXt101 for a device.

Run the following python script to export and optimize for iOS and Android:
```
python -m tetra_model_zoo.resnext101.export [ --help ]
```

# Model In-Application Deployment instructions
<a href="mailto:support@tetra.ai?subject=Request Access for Tetra Hub&body=Interest in using ResNeXt101 in model zoo for deploying on-device.">Get in touch with us</a> to learn more!

## License
- Code in this repository is covered by the LICENSE file at the repository root.
- ResNeXt101's license can be found [here](https://github.com/pytorch/vision/blob/main/LICENSE).

## References
* [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)
* [ResNeXt101 Source Repository](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
