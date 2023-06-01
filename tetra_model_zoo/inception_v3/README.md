[![Tetra AI](https://tetra.ai/img/logo.svg)](https://tetra.ai/)

# [InceptionNetV3: Imagenet classifier and general purpose backbone optimized for mobile and edge](https://tetraai.com/model-zoo/inception_v3)

InceptionNetV3 is a machine learning model that can classify images from the Imagenet dataset.
It can also be used as a backbone in building more complex models for specific use cases.
We present an optimized implementation of InceptionNetV3 suitable to export and run it on-device.

This is based on [InceptionNetV3](https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py). You can optionally
fine-tune the pre-trained model before walking through the examples below.

## Example & Usage
1. Install the package via pip:
```
pip install tetra_model_zoo[inception_v3]
```

2. Load the model & app
```
from tetra_model_zoo.inception_v3 import Model
from tetra_model_zoo.inception_v3 import App

app = App(Model.from_pretrained())
```

3. Run prediction
```
from tetra_model_zoo.utils.asset_loaders import load_image
from tetra_model_zoo.imagenet_classifier.test_utils import TEST_IMAGENET_IMAGE
from tetra_model_zoo.inception_v3 import MODEL_NAME

image = load_image(TEST_IMAGENET_IMAGE, MODEL_NAME)
app.predict(image)
```

See [app.py](../imagenet_classifier/app.py#L49) for more information about e2e usage of the model.

Please refer to our [general instructions on using models](../../#tetra-model-zoo)

### Optimize, Profile, and Validate InceptionNetV3 for a Device with Tetra Hub
Using Tetra Hub, you can easily optimize, profile, and validate InceptionNetV3 for a device.

Run the following python script to export and optimize for iOS and Android:
```
python -m tetra_model_zoo.inception_v3.export [ --help ]
```

# Model In-Application Deployment Instructions
<a href="mailto:support@tetra.ai?subject=Request Access for Tetra Hub&body=Interest in using InceptionV3 in model zoo for deploying on-device.">Get in touch with us</a> to learn more!

## License
- Code in this repository is covered by the LICENSE file at the repository root.
- InceptionNetV3's license can be found [here](https://github.com/pytorch/vision/blob/main/LICENSE).

## References
* [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567)
* [InceptionNetV3 Source Repository](https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py)
