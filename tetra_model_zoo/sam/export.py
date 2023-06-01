from __future__ import annotations

from typing import Any, Tuple

import torch
from torch.utils.mobile_optimizer import MobileOptimizerType, optimize_for_mobile

import tetra_hub as hub
from tetra_model_zoo.sam.model import (
    DEFAULT_MODEL_TYPE,
    SAMTetraWrapper,
    SegmentAnythingEncoder,
    SegmentAnythingONNXDecoder,
)
from tetra_model_zoo.utils.args import base_export_parser
from tetra_model_zoo.utils.hub import download_hub_models


def trace_sam(
    encoder: SegmentAnythingEncoder,
    decoder: SegmentAnythingONNXDecoder,
    input_img_size: Tuple[int, int],
    num_of_points: int,
) -> Tuple[Any, Any]:
    # Convert the model's encoder and decoder to pytorch traces.
    # Traces can be saved & loaded from disk.
    # With Tetra Hub, a pytorch trace can be exported to run efficiently on mobile devices!
    #
    # Returns: Tuple[Encoder Trace Object, Decoder Trace Object]
    #

    # Trace Encoder
    encoder_input_shape = encoder.get_input_spec(input_img_size)["image"][0]
    encoder_trace = torch.jit.trace(encoder, [torch.rand(encoder_input_shape)])

    # Trace Decoder
    # Optimize decoder to eliminate un-used parameters
    # and reduce traced model size.
    decoder_input_spec = decoder.get_input_spec(num_of_points)
    decoder_inputs = [torch.rand(v[0]) for v in decoder_input_spec.values()]
    decoder_trace = torch.jit.trace(decoder, decoder_inputs)
    decoder_trace = optimize_for_mobile(
        decoder_trace,
        optimization_blocklist={
            MobileOptimizerType.HOIST_CONV_PACKED_PARAMS,
            MobileOptimizerType.INSERT_FOLD_PREPACK_OPS,
            MobileOptimizerType.CONV_BN_FUSION,
        },
    )

    return encoder_trace, decoder_trace


def main():
    # Export parameters
    parser = base_export_parser()
    parser.add_argument(
        "--model_type",
        type=str,
        default=DEFAULT_MODEL_TYPE,
        help=f"SAM model type to load model. Default: {DEFAULT_MODEL_TYPE}.",
    )
    parser.add_argument(
        "--image_size",
        nargs="*",
        type=int,
        help="Space separated input image spatial dims to export models with. Default: 720 1280.",
    )
    parser.add_argument(
        "--num_of_points",
        type=int,
        default=1,
        help="Number of points to use for segmentation by decoder. Default: 1.",
    )
    args = parser.parse_args()

    input_img_size = (720, 1280) if args.image_size is None else tuple(args.image_size)
    if len(input_img_size) != 2:
        raise RuntimeError(
            "--image_size must have 2 space separated values for input image's spatial dims."
            f" Provided {len(input_img_size)} values."
        )

    # Load SAM models
    sam_tetra_wrapper = SAMTetraWrapper()

    # TODD: #4740 Export of SAM Encoder will be available soon
    sam_encoder = SegmentAnythingEncoder(sam_tetra_wrapper)
    sam_decoder = SegmentAnythingONNXDecoder(
        sam_tetra_wrapper,
        orig_img_size=input_img_size,
        single_mask_mode=(args.num_of_points == 1),
    )

    # Trace the model.
    _, traced_decoder = trace_sam(
        sam_encoder, sam_decoder, input_img_size, args.num_of_points
    )

    # Select the device you'd like to optimize for.
    devices = [hub.Device(device) for device in args.devices]

    # Submit the traced models for conversion & profiling.
    decoder_jobs = hub.submit_profile_job(
        name="Sam Decoder",
        model=traced_decoder,
        input_shapes=sam_decoder.get_input_spec(args.num_of_points),
        device=devices,
    )

    # Download the optimized assets!
    download_hub_models(decoder_jobs)


if __name__ == "__main__":
    main()
