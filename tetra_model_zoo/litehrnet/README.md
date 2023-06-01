[![Tetra AI](https://tetra.ai/img/logo.svg)](https://tetra.ai/)

# [LiteHRNet: Real-time human pose detection of images optimized for mobile and edge](https://tetraai.com/model-zoo/litehrnet)

LiteHRNet is a machine learning model that detects human pose and returns a location and confidence
for each of 17 joints.
We present an optimized implementation of LiteHRNet suitable to export for mobile applications.

This is based on [LiteHRNet](https://github.com/HRNet/Lite-HRNet). You can optionally
fine-tune the pre-trained model before walking through the examples below.

## Example & Usage
1. Install the package via pip:
```
pip install tetra_model_zoo[litehrnet]
```

2. Load the model & app
```
from tetra_model_zoo.litehrnet import Model
from tetra_model_zoo.litehrnet import App

app = App(Model.from_pretrained())
```

3. Run prediction
```
from tetra_model_zoo.utils.asset_loaders import load_image
from tetra_model_zoo.litehrnet.test import IMAGE_ADDRESS
from tetra_model_zoo.litehrnet import MODEL_NAME

image = load_image(IMAGE_ADDRESS, MODEL_NAME)
app.predict(image)
```

See [app.py](app.py#14) for more information about e2e usage of the model.

Please refer to our [general instructions on using models](../../#tetra-model-zoo)

### Optimize, Profile, and Validate LiteHRNet for a device with Tetra Hub
Using Tetra Hub, you can easily optimize, profile, and validate LiteHRNet for a device.

Run the following python script to export and optimize for iOS and Android:
```
python -m tetra_model_zoo.litehrnet.export [ --help ]
```

# Model In-Application Deployment instructions
<a href="mailto:support@tetra.ai?subject=Request Access for Tetra Hub&body=Interest in using LiteHRNet in model zoo for deploying on-device.">Get in touch with us</a> to learn more!

## License
- Code in this repository is covered by the LICENSE file at the repository root.
- LiteHRNet's license can be found [here](https://github.com/HRNet/Lite-HRNet/blob/hrnet/LICENSE).

## References
* [Lite-HRNet: A Lightweight High-Resolution Network](https://arxiv.org/abs/2104.06403)
* [LiteHRNet Source Repository](https://github.com/HRNet/Lite-HRNet)
* Demo image from [paper](https://arxiv.org/abs/2104.06403)
