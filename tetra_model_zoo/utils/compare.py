from typing import Union

import numpy as np
import torch


def compare_psnr(
    output_a: Union[torch.Tensor, np.ndarray],
    output_b: Union[torch.Tensor, np.ndarray],
    psnr_threshold: int,
    eps: float = 1e-5,
    eps2: float = 1e-10,
) -> None:
    """
    Computes the PSNR between two tensors.
    Returns True if its above the PSNR threshold otherwise False.
    """
    if not isinstance(output_a, np.ndarray):
        a = output_a.detach().numpy().flatten()
    else:
        a = output_a.flatten()
    if not isinstance(output_b, np.ndarray):
        b = output_b.detach().numpy().flatten()
    else:
        b = output_b.flatten()
    max_b = np.abs(b).max()
    sumdeltasq = 0.0
    sumdeltasq = ((a - b) * (a - b)).sum()
    sumdeltasq /= b.size
    sumdeltasq = np.sqrt(sumdeltasq)

    psnr = 20 * np.log10((max_b + eps) / (sumdeltasq + eps2))
    assert psnr > psnr_threshold
