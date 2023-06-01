from __future__ import annotations

import os
from typing import Any, List

import torch

from tetra_model_zoo.ddrnetslim.model import DDRNet
from tetra_model_zoo.utils.args import vision_export_parser


def trace(model: DDRNet, input_shape: List[int] = [1, 3, 1024, 2048]) -> Any:
    """
    Convert DDRNetSlim to a pytorch trace. Traces can be saved & loaded from disk.
    Returns: Trace Object
    """
    return torch.jit.trace(model, [torch.ones(input_shape)])


def main():
    import tetra_hub as hub
    from tetra_model_zoo.utils.hub import download_hub_models

    # Export parameters
    parser = vision_export_parser(
        default_x=1280, default_y=640, include_trace_option=True
    )
    parser.add_argument("--c", type=int, default=3, help="Number of image channels.")

    args = parser.parse_args()

    # Instantiate the model & a sample input.
    ddrnet_model = DDRNet.from_pretrained()

    # Trace the model.
    traced_ddrnet = trace(ddrnet_model, [args.b, args.c, args.y, args.x])

    if args.save_trace_and_exit:
        model_name = os.path.basename(args.weights).split(".")[0]
        model_path = os.path.join(os.getcwd(), f"{model_name}.torchscript.pt")
        torch.jit.save(traced_ddrnet, model_path)
        print(f"Saved torchscript to {model_path}")
        exit(0)

    # Select the device(s) you'd like to optimize for.
    devices = [hub.Device(x) for x in args.devices]

    # Submit the traced models for conversion & profiling.
    jobs = hub.submit_profile_job(
        name="ddrnetslim",
        model=traced_ddrnet,
        input_shapes=ddrnet_model.get_input_spec(image_size=(2048, 1024)),
        device=devices,
    )

    # Download the optimized assets!
    download_hub_models(jobs)


if __name__ == "__main__":
    main()
