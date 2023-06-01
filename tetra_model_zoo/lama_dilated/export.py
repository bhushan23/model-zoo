from __future__ import annotations

from typing import Any, List

import torch

import tetra_hub as hub
from tetra_model_zoo.lama_dilated.model import DEFAULT_WEIGHTS, LamaDilated
from tetra_model_zoo.utils.args import vision_export_parser
from tetra_model_zoo.utils.hub import download_hub_models


def trace(model: LamaDilated, input_shape: List[int]) -> Any:
    """
    Convert LamaDilated to a pytorch trace. Traces can be saved & loaded from disk.
    Returns: Trace Object
    """
    # Infer mask shape from input image shape.
    with torch.no_grad():
        image = torch.randn(
            (input_shape[0], input_shape[1], input_shape[2], input_shape[3])
        )
        mask = torch.randn((input_shape[0], 1, input_shape[2], input_shape[3]))
        model = torch.jit.trace(model, (image, mask))
        return model


def main():
    # Export parameters
    parser = vision_export_parser(
        default_x=512,
        default_y=512,
    )
    parser.add_argument("--c", type=int, default=3, help="Number of image channels.")

    args = parser.parse_args()

    # Instantiate the model & a sample input.
    model = LamaDilated.from_pretrained(DEFAULT_WEIGHTS)

    # Trace the model.
    traced_model = trace(model, [args.b, args.c, args.y, args.x])

    # Select the device(s) you'd like to optimize for.
    devices = [hub.Device(x) for x in args.devices]

    # Submit the traced models for conversion & profiling.
    jobs = hub.submit_profile_job(
        name="lama_dilated",
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
