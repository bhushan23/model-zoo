[![Tetra AI](https://tetra.ai/img/logo.svg)](https://tetra.ai/)

# [DDRNetSlim: Real-time Semantic Segmentation optimized for mobile and edge](https://tetraai.com/model-zoo/ddrnetslim)

DDRNet is a machine learning model that segments an image into semantic classes, specifically designed for road-based scenes. It is designed for the application of self-driving cars.
We present an optimized implementation of DDRNet (specifically, the DDRNet23Slim model) suitable to export for low latency mobile applications.

This is based on [DDRNet](https://github.com/chenjun2hao/DDRNet.pytorch). You can optionally
fine-tune the pre-trained model before walking through the examples below.

## Example & Usage
1. Install the package via pip:
```
pip install tetra_model_zoo[ddrnetslim]
```

2. Load the model & app
```
from tetra_model_zoo.ddrnetslim import Model
from tetra_model_zoo.ddrnetslim import App

app = App(Model.from_pretrained())
```

3. Run prediction
```
from tetra_model_zoo.utils.asset_loaders import load_image
from tetra_model_zoo.ddrnetslim.test import INPUT_IMAGE_ADDRESS
from tetra_model_zoo.ddrnetslim import MODEL_NAME

image = load_image(INPUT_IMAGE_ADDRESS, MODEL_NAME)
app.predict(image)
```

See [test.py](test.py) and [app.py](app.py#36) for more information about e2e usage of the model.

Please refer to our [general instructions on using models](../../#tetra-model-zoo)

### Optimize, Profile, and Validate DDRNet for a Device with Tetra Hub
Using Tetra Hub, you can easily optimize, profile, and validate DDRNet for a device.

Run the following python script to export and optimize for iOS and Android:
```
python -m tetra_model_zoo.ddrnetslim.export [ --help ]
```

# Model In-Application Deployment instructions
<a href="mailto:support@tetra.ai?subject=Request Access for Tetra Hub&body=Interest in using DDRNetSlim in model zoo for deploying on-device.">Get in touch with us</a> to learn more!

## License
- Code in this repository is covered by the LICENSE file at the repository root.
- DDRNet's license can be found [here](https://github.com/chenjun2hao/DDRNet.pytorch/blob/main/LICENSE).

## References
* [Deep Dual-resolution Networks for Real-time and
Accurate Semantic Segmentation of Road Scenes](https://arxiv.org/pdf/2101.06085.pdf)
* [DDRNet Source Repository](https://github.com/chenjun2hao/DDRNet.pytorch)
