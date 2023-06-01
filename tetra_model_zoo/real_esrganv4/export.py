from __future__ import annotations

import os
from typing import Any, List

import torch

import tetra_hub as hub
from tetra_model_zoo.real_esrgan.demo import WEIGHTS_HELP_MSG
from tetra_model_zoo.real_esrganv4.model import DEFAULT_WEIGHTS, RealESRGANv4
from tetra_model_zoo.utils.args import vision_export_parser
from tetra_model_zoo.utils.hub import download_hub_models


def trace(model: RealESRGANv4, input_shape: List[int] = [1, 3, 320, 320]) -> Any:
    """
    Convert RealESRGAN to a pytorch trace. Traces can be saved & loaded from disk.
    Returns: Trace Object
    """
    return torch.jit.trace(model, [torch.ones(input_shape)])


def main():
    # Export parameters
    parser = vision_export_parser(
        default_x=320, default_y=320, include_trace_option=True
    )
    parser.add_argument(
        "--weights", type=str, default=DEFAULT_WEIGHTS, help=WEIGHTS_HELP_MSG
    )
    parser.add_argument("--c", type=int, default=3, help="Number of image channels.")

    args = parser.parse_args()

    # Instantiate the model & a sample input.
    realesrgan_model = RealESRGANv4.from_pretrained(args.weights)

    # Trace the model.
    traced_realesrgan = trace(realesrgan_model, [args.b, args.c, args.x, args.y])

    if args.save_trace_and_exit:
        model_name = os.path.basename(args.weights).split(".")[0]
        model_path = os.path.join(os.getcwd(), f"{model_name}.torchscript.pt")
        torch.jit.save(traced_realesrgan, model_path)
        print(f"Saved torchscript to {model_path}")
        exit(0)

    # Select the device(s) you'd like to optimize for.
    devices = [hub.Device(x) for x in args.devices]

    # Submit the traced models for conversion & profiling.
    jobs = hub.submit_profile_job(
        name="realesrganv4",
        model=traced_realesrgan,
        input_shapes={"image": ((args.b, args.c, args.x, args.y), "float32")},
        device=devices,
    )

    # Download the optimized assets!
    download_hub_models(jobs)


if __name__ == "__main__":
    main()
