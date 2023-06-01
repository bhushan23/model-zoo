[![Tetra AI](https://tetra.ai/img/logo.svg)](https://tetra.ai/)

# [TrOCR: A transformer based OCR model optimized for Mobile](https://tetraai.com/model-zoo/trocr)

TrOCR is a transformer model that converts single-lines of writing in images to text output.
We present an optimized implementation of TrOCR suitable to export for low latency mobile applications.

This is based on [TrOCR small stage 1](https://huggingface.co/microsoft/trocr-small-stage1). You can optionally
fine-tune the pre-trained model on HuggingFace before walking through the examples below.

## Example & Usage

1. Install the package via pip:
```
pip install tetra_model_zoo[trocr]
```

2. Load the model & app
```
from tetra_model_zoo.trocr import Model
from tetra_model_zoo.trocr import App

app = App(Model.from_pretrained())
```

See [demo.py](demo.py) for sample usage of the model and app.

Please refer to our [general instructions on using models](../../#tetra-model-zoo)

### Optimize, Profile, and Validate TrOCR for a device with Tetra Hub
Using Tetra Hub, you can easily optimize, profile, and validate TrOCR for a device.

Run the following python script to export and optimize for iOS and Android:
```
python -m tetra_model_zoo.trocr.export [ --help ]
```

# Model In-Application Deployment instructions
<a href="mailto:support@tetra.ai?subject=Request Access for Tetra Hub&body=Interest in using TrOCR in model zoo for deploying on-device.">Get in touch with us</a> to learn more!

# Implementation Details
TrOCR is a transformer based model that consists of 2 parts (see `model.py`):
* Encoder --> Takes tensor (image) input, returns initial KV Cache for cross attention layers.
* Decoder --> Takes KV Cache and placeholder input token, returns predicted token.

A sample application (see `app.py (class TrOCRApp)`) ties these parts together to run TrOCR end-to-end.

## License
- Code in this repository is covered by the LICENSE file at the repository root.
- Microsoft TrOCR's license can be found [here](https://github.com/microsoft/unilm/blob/master/LICENSE).

## References
* [Whitepaper](https://arxiv.org/abs/2109.10282)
* [Huggingface Source Model](https://huggingface.co/microsoft/trocr-small-stage1)
