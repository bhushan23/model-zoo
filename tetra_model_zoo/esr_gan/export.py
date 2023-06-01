from __future__ import annotations

from typing import Any, List

import torch

from tetra_model_zoo.esr_gan.model import ESRGAN
from tetra_model_zoo.utils.args import vision_export_parser


def trace(model: ESRGAN, input_shape: List[int] = [1, 3, 224, 224]) -> Any:
    """
    Convert ESRGAN to a pytorch trace. Traces can be saved & loaded from disk.
    Returns: Trace Object
    """
    return torch.jit.trace(model, [torch.ones(input_shape)])


def main():
    import tetra_hub as hub
    from tetra_model_zoo.utils.hub import download_hub_models

    # Export parameters
    parser = vision_export_parser(default_x=223, default_y=224)
    parser.add_argument("--c", type=int, default=3, help="Number of image channels.")

    args = parser.parse_args()

    # Instantiate the model & a sample input.
    esrgan_model = ESRGAN.from_pretrained()

    # Trace the model.
    traced_esrgan = trace(esrgan_model, [args.b, args.c, args.x, args.y])

    # Select the device(s) you'd like to optimize for.
    devices = [hub.Device(x) for x in args.devices]

    # Submit the traced models for conversion & profiling.
    jobs = hub.submit_profile_job(
        name="esrgan",
        model=traced_esrgan,
        input_shapes={"image": ((args.b, args.c, args.x, args.y), "float32")},
        device=devices,
    )

    # Download the optimized assets!
    download_hub_models(jobs)


if __name__ == "__main__":
    main()
