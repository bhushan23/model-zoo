[![Tetra AI](https://tetra.ai/img/logo.svg)](https://tetra.ai/)

# [MediaPipe Hand: Real-time object detection optimized for mobile and edge](https://tetraai.com/model-zoo/mediapipe_hand)

The MediaPipe Hand Landmark Detector is a machine learning pipeline that predicts bounding boxes and pose skeletons of hands in an image.
We present an optimized implementation suitable to export for low latency mobile applications.

This is based on [MediaPipe PyTorch](https://github.com/zmurez/MediaPipePyTorch). You can optionally
fine-tune the pre-trained model before walking through the examples below.

## Example & Usage
1. Install the package via pip:
```
pip install tetra_model_zoo[mediapipe_hand]
```

2. Load the model & app
```
from tetra_model_zoo.mediapipe_hand import Model
from tetra_model_zoo.mediapipe_hand import App

# check demo.py for more details
app = App(Model.from_pretrained())
```

3. Run prediction
```
from tetra_model_zoo.utils.asset_loaders import load_image
from tetra_model_zoo.mediapipe_hand.test import INPUT_IMAGE_ADDRESS
from tetra_model_zoo.mediapipe_hand import MODEL_NAME

image = load_image(INPUT_IMAGE_ADDRESS, MODEL_NAME)
app.predict(image)
```

See [demo.py](demo.py) for more information about e2e usage of the model.

Please refer to our [general instructions on using models](../../#tetra-model-zoo)

### Optimize, Profile, and Validate MediaPipe Hands for a Device with Tetra Hub
Using Tetra Hub, you can easily optimize, profile, and validate the MediaPipe hand pipeline for a device.

Run the following python script to export and optimize for iOS and Android:
```
python -m tetra_model_zoo.mediapipe_hand.export [ --help ]
```
# Model In-Application Deployment instructions
<a href="mailto:support@tetra.ai?subject=Request Access for Tetra Hub&body=Interest in using MediaPipeHand in model zoo for deploying on-device.">Get in touch with us</a> to learn more!

## License
- Code in this repository is covered by the LICENSE file at the repository root.
- MediaPipe's license can be found [here](https://github.com/google/MediaPipe/blob/master/LICENSE).
- MediaPipe Pytorch's license can be found [here](https://github.com/zmurez/MediaPipePyTorch/blob/master/LICENSE)

## References
* [MediaPipe Hands: On-device Real-time Hand Tracking](https://arxiv.org/abs/2006.10214)
* [MediaPipe Repository](https://github.com/google/MediaPipe/)
* [MediaPipe PyTorch Repository](https://github.com/zmurez/MediaPipePyTorch/)
