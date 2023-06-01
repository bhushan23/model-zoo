[![Tetra AI](https://tetra.ai/img/logo.svg)](https://tetra.ai/)

# [RealESRGAN: Super-resolution of images optimised for mobile and edge](https://tetraai.com/model-zoo/real_esrgan)

![](https://tetra-public-assets.s3.us-west-2.amazonaws.com/model-zoo/realesrgan/v1/realesrgan_demo.jpg)
to
![](https://tetra-public-assets.s3.us-west-2.amazonaws.com/model-zoo/realesrgan/v1/realesrgan_demo_output.png)

Real-ESRGAN is a machine learning model that upscales an image with no loss in quality.
We present an optimized implementation of Real-ESRGAN suitable to export for mobile applications.
The implementation is a derivative of the Real-ESRGANv3 architecture, a smaller version
than the full Real-ESRGANx4 architecture.

This is based on [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN). You can optionally
fine-tune the pre-trained model before walking through the examples below.

## Example & Usage

1. Install the package via pip:
```
pip install tetra_model_zoo[real_esrgan]
```

2. Load the model & app
```
from tetra_model_zoo.real_esrgan import Model
from tetra_model_zoo.real_esrgan import App

# check demo.py for more details
app = App(Model.from_pretrained())
```

3. Run prediction
```
from tetra_model_zoo.utils.asset_loaders import load_image
from tetra_model_zoo.real_esrgan.test import IMAGE_ADDRESS
from tetra_model_zoo.real_esrgan import MODEL_NAME

image = load_image(IMAGE_ADDRESS, MODEL_NAME)
app.predict(image)
```

See [demo.py](demo.py) for more information about e2e usage of the model.

Please refer to our [general instructions on using models](../../#tetra-model-zoo)

### Optimize, Profile, and Validate RealESRGAN for a device with Tetra Hub
Using Tetra Hub, you can easily optimize, profile, and validate RealESRGAN for a device.

Run the following python script to export and optimize for iOS and Android:
```
python -m tetra_model_zoo.real_esrgan.export [ --help ]
```

# Model In-Application Deployment instructions
<a href="mailto:support@tetra.ai?subject=Request Access for Tetra Hub&body=Interest in using RealESRGAN in model zoo for deploying on-device.">Get in touch with us</a> to learn more!

## License
- Code in this repository is covered by the LICENSE file at the repository root.
- RealESRGAN's license can be found [here](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE).
- [Demo image](https://www.flickr.com/photos/birds_and_critters/53102982569/) is public domain dedication licensed

## References
* [Whitepaper](https://arxiv.org/abs/2107.10833)
* [RealESRGAN Source Repository](https://github.com/xinntao/Real-ESRGAN)
