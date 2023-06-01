from __future__ import annotations

import os
from typing import Any, List

import torch

import tetra_hub as hub
from tetra_model_zoo.utils.args import vision_export_parser
from tetra_model_zoo.utils.hub import download_hub_models
from tetra_model_zoo.yolov7.demo import WEIGHTS_HELP_MSG
from tetra_model_zoo.yolov7.model import YoloV7


def trace(model: YoloV7, input_shape: List[int] = [1, 3, 640, 640]) -> Any:
    """
    Convert YoloV7 to a pytorch trace. Traces can be saved & loaded from disk.
    Returns: Trace Object
    """
    return torch.jit.trace(model, [torch.ones(input_shape)])


def main():
    # Export parameters
    parser = vision_export_parser(
        default_x=640,
        default_y=640,
        dim_constraint=f"Must be a multiple of of {YoloV7.STRIDE_MULTIPLE}",
        include_trace_option=True,
    )
    parser.add_argument(
        "--weights", type=str, default="yolov7-tiny.pt", help=WEIGHTS_HELP_MSG
    )
    parser.add_argument("--c", type=int, default=3, help="Number of image channels.")

    args = parser.parse_args()

    # Instantiate the model & a sample input.
    yolo_model = YoloV7.from_pretrained(args.weights)

    # Trace the model.
    traced_yolo = trace(yolo_model, [args.b, args.c, args.y, args.x])

    if args.save_trace_and_exit:
        model_name = os.path.basename(args.weights).split(".")[0]
        model_path = os.path.join(os.getcwd(), f"{model_name}.torchscript.pt")
        torch.jit.save(traced_yolo, model_path)
        print(f"Saved torchscript to {model_path}")
        exit(0)

    # Select the device(s) you'd like to optimize for.
    devices = [hub.Device(x) for x in args.devices]

    # Submit the traced models for conversion & profiling.
    jobs = hub.submit_profile_job(
        name="yolov7",
        model=traced_yolo,
        input_shapes=yolo_model.get_input_spec(),
        device=devices,
    )

    # Download the optimized assets!
    download_hub_models(jobs)


if __name__ == "__main__":
    main()
