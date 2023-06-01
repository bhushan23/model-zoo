from typing import Dict, List, Tuple

import torch

# PyTorch trace doesn't capture the input specs. Hence we need an additional
# InputSpec (name -> (shape, type)) when submitting profiling job to TetraHub.
# This is a subtype of tetra_hub.InputSpecs
InputSpec = Dict[str, Tuple[Tuple[int, ...], str]]


def str_to_torch_dtype(s):
    return dict(
        int32=torch.int32,
        float32=torch.float32,
    )[s]


def make_torch_inputs(spec: InputSpec) -> List[torch.Tensor]:
    """Make sample torch inputs from input spec"""
    torch_input = []
    for sp in spec.values():
        torch_dtype = str_to_torch_dtype(sp[1])
        if sp[1] in {"int32"}:
            t = torch.randint(10, sp[0]).to(torch_dtype)
        else:
            t = torch.rand(sp[0]).to(torch_dtype)
        torch_input.append(t)
    return torch_input
