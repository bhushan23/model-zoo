[![Tetra AI](https://tetra.ai/img/logo.svg)](https://tetra.ai/)

# [Optimized CLIP](https://tetraai.com/model-zoo/optimized_clip)

CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image. This version of clip has been optimized using modules provided by [ANE Transformers](https://github.com/apple/ml-ane-transformers) repository provided by Apple.

## Example & Usage

1. Install the package via pip:
```
pip install tetra_model_zoo[optimized_clip]
```

2. Load the model & app
```
from tetra_model_zoo.optimized_clip import Model
from tetra_model_zoo.optimized_clip import App

app = App(Model.from_pretrained())
```

See [demo.py](demo.py) for more information about e2e usage of the model.

Please refer to our [general instructions on using models](../../#tetra-model-zoo)

### Optimize, Profile, and Validate CLIP for a device with Tetra Hub
Using Tetra Hub, you can easily optimize, profile, and validate the model for a device.

Run the following python script to export and optimize for iOS and Android:
```
python -m tetra_model_zoo.ane_clip.export [--help]
```

# Model In-Application Deployment instructions
<a href="mailto:support@tetra.ai?subject=Request Access for Tetra Hub&body=Interest in using CLIP (optimized) in model zoo for deploying on-device.">Get in touch with us</a> to learn more!

## License
- Code in this repository is covered by the LICENSE file at the repository root.
- OpenAI Clip's license can be found [here](https://github.com/openai/CLIP/blob/main/LICENSE).
- ANETransformer's license can be found [here](https://github.com/apple/ml-ane-transformers/blob/main/LICENSE.md)

## References
* [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
* [Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers)
* [Open AI CLIP's Repository](https://github.com/openai/CLIP)
* [Apple's ANE Transformers Repository](https://github.com/apple/ml-ane-transformers/)
