from __future__ import annotations

from typing import Any, List

import torch

import tetra_hub as hub
from tetra_model_zoo.aotgan.model import AOTGAN, DEFAULT_WEIGHTS, WEIGHTS_HELP_MSG
from tetra_model_zoo.utils.args import vision_export_parser
from tetra_model_zoo.utils.hub import download_hub_models


def trace(model: AOTGAN, input_shape: List[int] = [1, 3, 512, 512]) -> Any:
    """
    Convert AOTGAN to a pytorch trace. Traces can be saved & loaded from disk.
    Returns: Trace Object
    """
    # Infer mask shape from input image shape.
    B, _, H, W = input_shape
    return torch.jit.trace(model, [torch.ones(input_shape), torch.ones(B, 1, H, W)])


def main():
    # Export parameters
    parser = vision_export_parser(
        default_x=640,
        default_y=640,
    )
    parser.add_argument(
        "--weights", type=str, default=DEFAULT_WEIGHTS, help=WEIGHTS_HELP_MSG
    )
    parser.add_argument("--c", type=int, default=3, help="Number of image channels.")

    args = parser.parse_args()

    # Instantiate the model & a sample input.
    model = AOTGAN.from_pretrained(args.weights)

    # Trace the model.
    traced_model = trace(model, [args.b, args.c, args.y, args.x])

    # Select the device(s) you'd like to optimize for.
    devices = [hub.Device(x) for x in args.devices]

    # Submit the traced models for conversion & profiling.
    jobs = hub.submit_profile_job(
        name="aotgan",
        model=traced_model,
        input_shapes=model.get_input_spec(
            image_size=[args.y, args.x], batch_size=args.b, num_channels=args.c
        ),
        device=devices,
    )

    # Download the optimized assets!
    download_hub_models(jobs)


if __name__ == "__main__":
    main()
