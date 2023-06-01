[![Tetra AI](https://tetra.ai/img/logo.svg)](https://tetra.ai/)

# [Whisper: OpenAI Whisper optimized for mobile and edge](https://tetraai.com/model-zoo/whisper_asr)

Whisper is a state of art speech recognition model from OpenAI [Whisper model](https://github.com/openai/whisper/tree/main). We present a version that's optimized for mobile and edge inference with the Tetra Platform. TetraWhisper-LargeV2 can transcribe 1 minute of audio in about X seconds when running on M1.


## Example & Usage

1. Install the package via pip:
```
pip install tetra_model_zoo[whisper_asr]
```

2. Load the model & app
```
from tetra_model_zoo.whisper_asr import Model
from tetra_model_zoo.whisper_asr import App

app = App(Model.from_pretrained())
```

See [demo.py](demo.py) for sample usage of the model and app.

Please refer to our [general instructions on using models](../../#tetra-model-zoo)

### Optimize, Profile, and Validate Whisper for a device with Tetra Hub
Using Tetra Hub, you can easily optimize, profile, and validate Whisper for a device.

Run the following python script to export and optimize for iOS and Android:
```
python -m tetra_model_zoo.whisper_asr.export [ --help ]
```

# Model In-Application Deployment instructions
<a href="mailto:support@tetra.ai?subject=Request Access for Tetra Hub&body=Interest in using Whisper in model zoo for deploying on-device.">Get in touch with us</a> to learn more!


## License
- Code in this repository is covered by the LICENSE file at the repository root.


## References
* [Whisper paper](https://cdn.openai.com/papers/whisper.pdf)
* [Official whisper repo](https://github.com/openai/whisper/tree/main)
