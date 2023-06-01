from __future__ import annotations

from typing import Any, Tuple

import torch

import tetra_hub as hub
from tetra_model_zoo.trocr.model import (
    HUGGINGFACE_TROCR_MODEL,
    TrOCR,
    TrOCRDecoder,
    TrOCREncoder,
)
from tetra_model_zoo.utils.args import base_export_parser
from tetra_model_zoo.utils.hub import download_hub_models


def trace_trocr(encoder: TrOCREncoder, decoder: TrOCRDecoder) -> Tuple[Any, Any]:
    # Convert the model's encoder and decoder to pytorch traces. Traces can be saved & loaded from disk.
    # With Tetra Hub, a pytorch trace can be exported to run efficiently on mobile devices!
    #
    # Returns: Tuple[Encoder Trace Object, Decoder Trace Object]
    #

    # Trace Encoder + KV Generator
    encoder_input_shape = encoder.get_input_spec()["pixel_values"][0]
    encoder_trace = torch.jit.trace(encoder, [torch.rand(encoder_input_shape)])

    # Trace Decoder
    decoder_input_shapes = decoder.get_input_spec()
    decoder_inputs = []
    for _, v in decoder_input_shapes.items():
        decoder_inputs.append(torch.rand(v[0]))
    decoder_inputs[0] = decoder_inputs[0].type(torch.int32)
    decoder_trace = torch.jit.trace(decoder, decoder_inputs)

    return encoder_trace, decoder_trace


def main():
    # Export parameters
    parser = base_export_parser()
    parser.add_argument(
        "--hf_trocr_model",
        type=str,
        default=HUGGINGFACE_TROCR_MODEL,
        help=f"huggingface model to load. Tested with {HUGGINGFACE_TROCR_MODEL}.",
    )
    args = parser.parse_args()

    # Load Huggingface source
    model = TrOCR.from_pretrained(args.hf_trocr_model)
    encoder = model.encoder
    decoder = model.decoder

    # Trace the model.
    traced_encoder, traced_decoder = trace_trocr(encoder, decoder)  # type: ignore

    # Select the device you'd like to optimize for.
    devices = [hub.Device(device) for device in args.devices]

    # Submit the traced models for conversion & profiling.
    encoder_jobs = hub.submit_profile_job(
        name="trocr_encoder",
        model=traced_encoder,
        input_shapes=encoder.get_input_spec(),
        device=devices,
    )
    decoder_jobs = hub.submit_profile_job(
        name="trocr_decoder",
        model=traced_decoder,
        input_shapes=decoder.get_input_spec(),
        device=devices,
    )

    # Download the optimized assets!
    print("Exported encoder(s):\n")
    download_hub_models(encoder_jobs)
    print("Exported decoder(s):\n")
    download_hub_models(decoder_jobs)


if __name__ == "__main__":
    main()
