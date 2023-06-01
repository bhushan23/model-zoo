from __future__ import annotations

from typing import Any, List

import torch

import tetra_hub as hub
from tetra_model_zoo.litehrnet.model import LiteHRNet
from tetra_model_zoo.utils.args import vision_export_parser
from tetra_model_zoo.utils.hub import download_hub_models


def trace(model: LiteHRNet, input_shape: List[int] = [3, 256, 192]) -> Any:
    """
    Convert LiteHRNet to a pytorch trace. Traces can be saved & loaded from disk.
    Returns: Trace Object
    """
    return torch.jit.trace(model, [torch.ones(input_shape)])


def main():
    # Export parameters
    parser = vision_export_parser(
        default_x=256, default_y=192, include_trace_option=True
    )
    parser.add_argument("--c", type=int, default=3, help="Number of image channels.")

    args = parser.parse_args()

    # Instantiate the model & a sample input.
    litehrnet_model = LiteHRNet.from_pretrained()

    # Trace the model.
    traced_litehrnet = trace(litehrnet_model, [args.b, args.c, args.x, args.y])

    # Select the device(s) you'd like to optimize for.
    devices = [hub.Device(x) for x in args.devices]

    # Submit the traced models for conversion & profiling.
    jobs = hub.submit_profile_job(
        name="litehrnet",
        model=traced_litehrnet,
        input_shapes=litehrnet_model.get_input_spec((args.x, args.y), args.b, args.c),
        device=devices,
    )

    # Download the optimized assets!
    download_hub_models(jobs)


if __name__ == "__main__":
    main()
