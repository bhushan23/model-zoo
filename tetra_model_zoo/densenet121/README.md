[![Tetra AI](https://tetra.ai/img/logo.svg)](https://tetra.ai/)

# [Densenet: Imagenet classifier and general purpose backbone optimized for mobile and edge](https://tetraai.com/model-zoo/densenet121)

Densenet is a machine learning model that can classify images from the Imagenet dataset.
It can also be used as a backbone in building more complex models for specific use cases.
We present an optimized implementation of Densenet suitable to export and run it on-device.

This is based on [Densenet](https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py). You can optionally
fine-tune the pre-trained model before walking through the examples below.

## Example & Usage
1. Install the package via pip:
```
pip install tetra_model_zoo[densenet121]
```

2. Load the model & app
```
from tetra_model_zoo.densenet121 import Model
from tetra_model_zoo.densenet121 import App

app = App(Model.from_pretrained())
```

3. Run prediction
```
from tetra_model_zoo.utils.asset_loaders import load_image
from tetra_model_zoo.imagenet_classifier.test_utils import TEST_IMAGENET_IMAGE
from tetra_model_zoo.densenet121 import MODEL_NAME

image = load_image(TEST_IMAGENET_IMAGE, MODEL_NAME)
app.predict(image)
```

See [app.py](../imagenet_classifier/app.py#L49) for more information about e2e usage of the model.

Please refer to our [general instructions on using models](../../#tetra-model-zoo)

### Optimize, Profile, and Validate Densenet for a Device with Tetra Hub
Using Tetra Hub, you can easily optimize, profile, and validate  for a device.

Run the following python script to export and optimize for iOS and Android:
```
python -m tetra_model_zoo.densenet121.export [ --help ]
```

# Model In-Application Deployment Instructions
<a href="mailto:support@tetra.ai?subject=Request Access for Tetra Hub&body=Interest in using DenseNet121 in model zoo for deploying on-device.">Get in touch with us</a> to learn more!

## License
- Code in this repository is covered by the LICENSE file at the repository root.
- Densenet's license can be found [here](https://github.com/pytorch/vision/blob/main/LICENSE).

## References
* [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
* [Densenet Source Repository](https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py)
