[![Tetra AI](https://tetra.ai/img/logo.svg)](https://tetra.ai/)

# [OpenAI CLIP](https://tetraai.com/model-zoo/openai_clip)

CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image.

## Example & Usage

1. Install the package via pip:
```
pip install tetra_model_zoo[openai_clip]
```

2. Load the model & app
```
from tetra_model_zoo.openai_clip import Model
from tetra_model_zoo.openai_clip import App

# check demo.py for more details
app = App(Model.from_pretrained())
```

See [demo.py](demo.py) for more information about e2e usage of the model.

Please refer to our [general instructions on using models](../../#tetra-model-zoo)

### Optimize, Profile, and Validate CLIP for a device with Tetra Hub
Using Tetra Hub, you can easily optimize, profile, and validate the model for a device.

Run the following python script to export and optimize for iOS and Android:
```
python -m tetra_model_zoo.openai_clip.export [--help]
```

# Model In-Application Deployment instructions
<a href="mailto:support@tetra.ai?subject=Request Access for Tetra Hub&body=Interest in using OpenAI CLIP in model zoo for deploying on-device.">Get in touch with us</a> to learn more!

## License
- Code in this repository is covered by the LICENSE file at the repository root.
- OpenAI CLIP's license can be found [here](https://github.com/openai/CLIP/blob/main/LICENSE).

## References
* [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
* [Repository](https://github.com/openai/CLIP)
