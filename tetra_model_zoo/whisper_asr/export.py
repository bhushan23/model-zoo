import os

import torch

import tetra_hub as hub
from tetra_model_zoo.utils.input_spec import make_torch_inputs
from tetra_model_zoo.whisper_asr.model import Whisper

if __name__ == "__main__":
    # For other model sizes, see https://github.com/openai/whisper/blob/main/whisper/__init__.py#L17
    model_version = "tiny.en"
    model = Whisper.from_pretrained(model_version)
    encoder = model.encoder
    decoder = model.decoder

    # Trace encoder
    torch_inputs = make_torch_inputs(encoder.get_input_spec())
    encoder_trace = torch.jit.trace(encoder, torch_inputs)

    # Trace decoder
    torch_inputs = make_torch_inputs(decoder.get_input_spec())
    decoder_trace = torch.jit.trace(decoder, torch_inputs)
    # Select the device you'd like to optimize for.
    devices = [
        hub.Device("Apple iPhone 14 Pro"),
        hub.Device("Samsung Galaxy S23 Ultra"),
    ]

    # Submit the traced models for optimization and profiling.
    encoder_jobs = hub.submit_profile_job(
        name=f"whisper_{model_version}_encoder",
        model=encoder_trace,
        input_shapes=encoder.get_input_spec(),
        device=devices,
    )

    decoder_jobs = hub.submit_profile_job(
        name=f"whisper_{model_version}_decoder",
        model=decoder_trace,
        input_shapes=decoder.get_input_spec(),
        device=devices,
    )

    # Download the optimized assets!
    encoder_model_paths = []
    for job in encoder_jobs:
        encoder_model_paths.append(job.download_target_model(os.getcwd()))
    decoder_model_paths = []
    for job in decoder_jobs:
        decoder_model_paths.append(job.download_target_model(os.getcwd()))

    print("Exported encoder(s):\n" + "\n".join(encoder_model_paths))
    print("Exported decoder(s):\n" + "\n".join(decoder_model_paths))
