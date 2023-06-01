"""
Utility Functions for parsing input args for export and other customer facing scripts.
"""

import argparse
from typing import Optional


def base_export_parser(include_trace_option: bool = False) -> argparse.ArgumentParser:
    """
    Base arg parser that specifies input args for an export script.

    Parameters:
        include_trace_options: includes saving trace option if set.

    Returns:
        Arg parser object.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["Apple iPhone 14 Pro", "Samsung Galaxy S23 Ultra"],
        help="Device[s] to export to.",
    )
    if include_trace_option:
        parser.add_argument(
            "--save_trace_and_exit",
            action="store_true",
            help="Write torchscript to current directory and exits.",
        )
    return parser


def vision_export_parser(
    default_x: int,
    default_y: int,
    dim_constraint: Optional[str] = None,
    include_trace_option: bool = False,
) -> argparse.ArgumentParser:
    """
    Argument parser for a vision model's export script.
    Takes input image dimensions as args.

    Parameters:
        default_x: Default width in pixels.
        default_y: Default height in pixels.
        dim_constraint: Help message stating any constraints on the input dimensions.
        include_trace_options: includes saving trace option if set.

    Returns:
        Arg parser object.
    """
    parser = base_export_parser(include_trace_option=include_trace_option)
    dim_constraint = dim_constraint or ""
    parser.add_argument(
        "--x",
        type=int,
        default=default_x,
        help=f"Input image width. {dim_constraint}",
    )
    parser.add_argument(
        "--y",
        type=int,
        default=default_y,
        help=f"Input image height. {dim_constraint}",
    )
    parser.add_argument("--b", type=int, default=1, help="Batch size.")
    return parser
