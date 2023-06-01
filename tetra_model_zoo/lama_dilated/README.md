[![Tetra AI](https://tetra.ai/img/logo.svg)](https://tetra.ai/)

# [LaMa-Dilated: High resolution image in-painting optimized for mobile and edge](https://tetraai.com/model-zoo/lama_dilated)

LaMa-Dilated is a machine learning model that allows to erase and in-paint part of given input image.
We present an optimized implementation of AOT-GAN suitable to export and run it on-device.

This is based on [LaMa-for-inpainting](https://github.com/advimman/lama/tree/main). You can optionally
fine-tune the pre-trained model before walking through the examples below.

## Example & Usage
1. Install the package via pip:
```
pip install tetra_model_zoo[lama_dilated]
```

2. Load the model & app
```
from tetra_model_zoo.lama_dilated import Model
from tetra_model_zoo.lama_dilated import App

app = App(Model.from_pretrained())
```

3. Run prediction
```
from tetra_model_zoo.utils.asset_loaders import load_image
from tetra_model_zoo.lama_dilated.test import IMAGE_ADDRESS
from tetra_model_zoo.lama_dilated import MODEL_NAME

image = load_image(IMAGE_ADDRESS, MODEL_NAME)
app.predict(image)
```

See [app.py](../imagenet_classifier/app.py#L49) for more information about e2e usage of the model.

Please refer to our [general instructions on using models](../../#tetra-model-zoo)


See [app.py](../repaint/app.py#L28) for example use of LaMa-Dilated for inpainting.

### Optimize, Profile, and Validate LaMa Dilated for a Device with Tetra Hub
Using Tetra Hub, you can easily optimize, profile, and validate LaMa Dilated for a device.

Run the following python script to export and optimize for iOS and Android:
```
python -m tetra_model_zoo.lama_dilated.export [ --help ]
```

# Model In-Application Deployment Instructions
<a href="mailto:support@tetra.ai?subject=Request Access for Tetra Hub&body=Interest in using LaMaDilated in model zoo for deploying on-device.">Get in touch with us</a> to learn more!

## License
- Code in this repository is covered by the LICENSE file at the repository root.
- LaMa Dilated's license can be found [here](https://github.com/advimman/lama/blob/main/LICENSE).

## References
* [Resolution-robust Large Mask Inpainting with Fourier Convolutions](https://arxiv.org/abs/2109.07161)
* [Source Repository](https://github.com/advimman/lama)
