import os
from typing import List, Union

import tetra_hub as hub


def download_hub_models(jobs: Union[hub.Job, List[hub.Job]]) -> List[str]:
    """
    Downloads compiled models from a submitted profiling job.

    Parameters:
        jobs: tetra_hub.Job objects that resulted from profiling.
    Returns:
        List of local paths to which models were downloaded.
    """

    if not isinstance(jobs, list):
        jobs = [jobs]

    model_paths = []
    for j in jobs:
        assert isinstance(j, hub.ProfileJob)
        model_path = j.download_target_model(os.getcwd())
        model_paths.append(model_path)
        print(f"Exported model: {model_path}\n")
    return model_paths
