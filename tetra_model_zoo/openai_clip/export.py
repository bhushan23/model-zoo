from __future__ import annotations

import torch

import tetra_hub as hub
from tetra_model_zoo.openai_clip.model import Clip
from tetra_model_zoo.utils.args import base_export_parser
from tetra_model_zoo.utils.hub import download_hub_models


def trace_clip(clip_model: Clip):
    # Trace model
    text_encoder = clip_model.text_encoder
    image_encoder = clip_model.image_encoder
    text_shape = text_encoder.get_input_spec()
    image_shape = image_encoder.get_input_spec()
    text = torch.randint(
        size=text_shape["text"][0], high=22000, low=0, dtype=torch.int32
    )
    image = torch.rand(image_shape["image"][0])
    clip_text_encoder_trace = torch.jit.trace(text_encoder, text)
    clip_image_encoder_trace = torch.jit.trace(image_encoder, image)
    return clip_text_encoder_trace, clip_image_encoder_trace, text_shape, image_shape


def main():
    parser = base_export_parser()
    args = parser.parse_args()

    # Instantiate the model & a sample input.
    clip_model = Clip.from_pretrained()
    # Trace the model.
    (
        clip_text_encoder_trace,
        clip_image_encoder_trace,
        text_shape,
        image_shape,
    ) = trace_clip(clip_model)

    # Select the device(s) you'd like to optimize for.
    devices = [hub.Device(x) for x in args.devices]

    # Submit the traced models for conversion & profiling.
    text_encoder_jobs = hub.submit_profile_job(
        name="openai_clip_text",
        model=clip_text_encoder_trace,
        input_shapes=text_shape,
        device=devices,
    )

    image_encoder_jobs = hub.submit_profile_job(
        name="openai_clip_image",
        model=clip_image_encoder_trace,
        input_shapes=image_shape,
        device=devices,
    )

    # Download the optimized asset!
    download_hub_models(text_encoder_jobs)
    download_hub_models(image_encoder_jobs)


if __name__ == "__main__":
    main()
