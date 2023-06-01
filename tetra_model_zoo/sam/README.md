[![Tetra AI](https://tetra.ai/img/logo.svg)](https://tetra.ai/)

# [SAM: Segment Anything with real-time segmentation decoder optimized for mobile and edge](https://tetraai.com/model-zoo/sam)

SegmentAnything is a Transformer based model for image segmentation.
This model follows encoder-decoder architecture:
  - Large and heavy image encoder to generate image embeddings
  - Light-weight decoder to work on image embedding for point and mask based segmentation

This model is a great example of using cloud and edge together where large model runs over the cloud and light weight model runs on the edge and performs multiple segmentations locally. We present both, Encoder (suitable for cloud deployment) and Decoder (suitable for low latency mobile/edge application).

This is based on [segment-anything](https://github.com/tetraai/segment-anything). You can optionally
fine-tune the pre-trained model before walking through the examples below.

## Example & Usage

1. Install the package via pip:
```
pip install tetra_model_zoo[sam]
```

2. Load the model & app
```
from tetra_model_zoo.sam import Model
from tetra_model_zoo.sam import App

app = App(Model.from_pretrained())
```

3. Run prediction
See [demo.py](demo.py#L62) for more information about e2e usage of the model.

Please refer to our [general instructions on using models](../../#tetra-model-zoo)

### Optimize, Profile, and Validate SAM for a device with Tetra Hub
Using Tetra Hub, you can easily optimize, profile, and validate SAM for a device.

Run the following python script to export and optimize for iOS and Android:
```
python -m tetra_model_zoo.sam.export [ --help ]
```

# Model In-Application Deployment instructions
<a href="mailto:support@tetra.ai?subject=Request Access for Tetra Hub&body=Interest in using SAM in model zoo for deploying on-device.">Get in touch with us</a> to learn more!

## License
- Code in this repository is covered by the LICENSE file at the repository root.
- segment-anything's license can be found [here](https://github.com/facebookresearch/segment-anything/blob/main/LICENSE).

## References
* [Segment Anything](https://arxiv.org/abs/2304.02643)
* [segment-anything Source Repository](https://github.com/facebookresearch/segment-anything)
