import os
from typing import List, Union

import torch


def display_with_sig_figs(num: float, num_sig_figs: int = 3) -> str:
    """
    Displays the given number as a string with the appropriate number of
    significant figures. Example:

        display_with_sig_figs(1234.2, num_sig_figs=3) -> "1230"

    Parameters:
        num: Number to display.
        num_sig_figs: How many sig figs to use.
    """
    rounded_num = float(f"{num:.{num_sig_figs}g}")
    num_digits = len(str(int(rounded_num)))

    # Only display as many numbers after the decimal point to fit number of sig figs
    return f"{rounded_num:.{max(0, num_sig_figs - num_digits)}f}"


def get_formatted_size(size: float, units: List[str], unit_step_size: float) -> str:
    """
    Formats the number according to the units provided. For example:

    format_size(3600, units=["B", "KB", ...], unit_step_size=1024.0)

    would return "3.6KB"

    Parameters:
        num: Raw count of size.
        units: A list of increasing unit sizes (e.g. ["B", "KB", ...])
        unit_step_size: The ratio in size between successive units.
    """

    unit_index = 0

    while size >= unit_step_size and unit_index < len(units) - 1:
        size /= unit_step_size
        unit_index += 1

    return f"{display_with_sig_figs(size)}{units[unit_index]}"


def get_num_trainable_parameters(
    model: torch.nn.Module, as_str: bool = True
) -> Union[str, int]:
    """
    Computes the number of trainable parameters in the model.

    Parameters:
        model: The model of which to count parameters.
        as_str: Whether to return the result as an int or a string formatted to 2 sig figs.
    """
    parameter_cnt = 0
    for param in model.parameters():
        if param.requires_grad:
            parameter_cnt += param.numel()
    if not (as_str):
        return parameter_cnt
    return get_formatted_size(parameter_cnt, ["", "K", "M", "B", "T"], 1000.0)


def get_checkpoint_file_size(model_path: str, as_str: bool = True) -> Union[str, int]:
    """
    Computes how much memory the model checkpoint consumes.

    Parameters:
        model_path: Path to the model checkpoint file.
        as_str: Whether to return the result as an int or a string formatted to 2 sig figs.
    """
    num_bytes = os.path.getsize(model_path)

    if not (as_str):
        return num_bytes

    return get_formatted_size(num_bytes, [" B", " KB", " MB", " GB", " TB"], 1024.0)
