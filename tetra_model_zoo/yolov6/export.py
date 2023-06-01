from __future__ import annotations

from typing import Any, List

import torch

import tetra_hub as hub
from tetra_model_zoo.utils.args import vision_export_parser
from tetra_model_zoo.utils.hub import download_hub_models
from tetra_model_zoo.yolov6.model import DEFAULT_WEIGHTS, YoloV6

WEIGHTS_HELP_MSG = "Specify checkpoint `.pth` name from https://github.com/meituan/YOLOv6/releases/tag/0.4.0"


def trace(model: YoloV6, input_shape: List[int] = [1, 3, 640, 640]) -> Any:
    """
    Convert YoloV6 to a pytorch trace. Traces can be saved & loaded from disk.
    Returns: Trace Object
    """
    return torch.jit.trace(model, [torch.ones(input_shape)])


def main():
    # Export parameters
    parser = vision_export_parser(
        default_x=640,
        default_y=640,
        include_trace_option=True,
    )
    parser.add_argument(
        "--weights", type=str, default=DEFAULT_WEIGHTS, help=WEIGHTS_HELP_MSG
    )
    parser.add_argument("--c", type=int, default=3, help="Number of image channels.")

    args = parser.parse_args()

    # Instantiate the model & a sample input.
    model = YoloV6.from_pretrained(args.weights)

    # Trace the model.
    traced_model = trace(model, [args.b, args.c, args.y, args.x])

    # Select the device(s) you'd like to optimize for.
    devices = [hub.Device(x) for x in args.devices]

    # Submit the traced models for conversion & profiling.
    jobs = hub.submit_profile_job(
        name="yolov6_e2e",
        model=traced_model,
        input_shapes=model.get_input_spec([args.y, args.x], args.b, args.c),
        device=devices,
    )

    # Download the optimized assets!
    download_hub_models(jobs)


if __name__ == "__main__":
    main()
