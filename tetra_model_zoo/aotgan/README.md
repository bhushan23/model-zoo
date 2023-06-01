[![Tetra AI](https://tetra.ai/img/logo.svg)](https://tetra.ai/)

# [AOT-GAN: High resolution image in-painting optimized for mobile and edge](https://tetraai.com/model-zoo/aotgan)

AOT-GAN is a machine learning model that allows to erase and in-paint part of given input image.
We present an optimized implementation of AOT-GAN suitable to export and run it on-device.

This is based on [AOT-GAN-for-inpainting](https://github.com/researchmm/AOT-GAN-for-Inpainting). You can optionally
fine-tune the pre-trained model before walking through the examples below.

## Example and Usage
1. Load the model & app
```
from tetra_model_zoo.aotgan import Model
from tetra_model_zoo.aotgan import App

app = App(Model.from_pretrained())
```

2. Run prediction
```
from tetra_model_zoo.utils.asset_loaders import load_image
from tetra_model_zoo.aotgan.test import IMAGE_ADDRESS, MASK_ADDRESS
from tetra_model_zoo.aotgan import MODEL_NAME

image = load_image(IMAGE_ADDRESS, MODEL_NAME)
mask_image = load_image(MASK_ADDRESS, MODEL_NAME)
app.predict(image, mask_image)
```

See [app.py](../repaint/app.py#L28) for more example use of AOT-GAN for inpainting.

Please refer to our [general instructions on using models](../../#tetra-model-zoo)

### Optimize, Profile, and Validate AOT-GAN for a Device with Tetra Hub
Using Tetra Hub, you can easily optimize, profile, and validate AOT-GAN for a device.

Run the following python script to export and optimize for iOS and Android:
```
python -m tetra_model_zoo.aotgan.export [ --help ]
```

# Model In-Application Deployment Instructions
<a href="mailto:support@tetra.ai?subject=Request Access for Tetra Hub&body=Interest in using AOTGAN in model zoo for deploying on-device.">Get in touch with us</a> to learn more!

## License
- Code in this repository is covered by the LICENSE file at the repository root.
- The source AOT-GAN repository does not provide a license for their code.

## References
* [Aggregated Contextual Transformations for High-Resolution Image Inpainting](https://arxiv.org/pdf/2104.01431.pdf)
* [AOT-GAN Source Repository](https://github.com/researchmm/AOT-GAN-for-Inpainting)
