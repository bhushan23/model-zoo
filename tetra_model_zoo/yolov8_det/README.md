[![Tetra AI](https://tetra.ai/img/logo.svg)](https://tetra.ai/)

# [YoloV8 Detector: Real-time object detection optimized for mobile and edge](https://tetraai.com/model-zoo/yolov8_det)

YoloV8 object detection is a machine learning model that predicts bounding boxes and classes of objects in an image.
We present an optimized implementation of YoloV6 suitable to export for low latency mobile applications.

This is based on [YOLOv8](https://docs.ultralytics.com/tasks/detect/). You can optionally
fine-tune the pre-trained model before walking through the examples below.

## Example & Usage

1. Install the package via pip:
```
pip install tetra_model_zoo[yolov8_det]
```

2. Load the model & app
```
from tetra_model_zoo.yolov8_det import Model
from tetra_model_zoo.yolov8_det import App

app = App(Model.from_pretrained())
```

3. Run prediction
```
from tetra_model_zoo.utils.asset_loaders import load_image
from tetra_model_zoo.yolov8_det.test import IMAGE_ADDRESS
from tetra_model_zoo.yolov8_det import MODEL_NAME

image = load_image(IMAGE_ADDRESS, MODEL_NAME)
app.predict(image)
```

See [app.py](../yolo/app.py#L73) for more information about e2e usage of the model.

Please refer to our [general instructions on using models](../../#tetra-model-zoo)

### Optimize, Profile, and Validate YOLOv8 for a device with Tetra Hub
Using Tetra Hub, you can easily optimize, profile, and validate YOLOv8 for a device.

Run the following python script to export and optimize for iOS and Android:
```
python -m tetra_model_zoo.yolov8_det.export [ --help ]
```

# Model In-Application Deployment instructions
<a href="mailto:support@tetra.ai?subject=Request Access for Tetra Hub&body=Interest in using YOLOv8 in model zoo for deploying on-device.">Get in touch with us</a> to learn more!

## License
- Code in this repository is covered by the LICENSE file at the repository root.
- YoloV6's license can be found [here](https://github.com/meituan/YOLOv8/blob/47625514e7480706a46ff3c0cd0252907ac12f22/LICENSE).

## References
* [Real-Time Flying Object Detection with YOLOv8](https://arxiv.org/abs/2305.09972)
* [YoloV8 Source Repository](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/models/yolo/detect)
* [YoloV8 Ultralytics Repository](https://docs.ultralytics.com/tasks/detect)
