#!/usr/bin/env python
#
# Copyright (c) 2022, Tetra Intelligence Systems Inc. All rights reserved.
#

import pathlib
from typing import Dict

from setuptools import find_packages, setup

r_file = "requirements.txt"

model_zoo_path = pathlib.Path(__file__).parent / "tetra_model_zoo"
requirements_path = model_zoo_path / r_file

version_path = model_zoo_path / "_version.py"
version_locals: Dict[str, str] = {}
exec(open(version_path).read(), version_locals)

description = "Models optimized for export to run on device."
setup(
    name="tetra_model_zoo",
    version=version_locals["__version__"],
    description=description,
    long_description=description,
    author="Tetra Intelligence Systems Inc.",
    url="https://tetra.ai/",
    packages=find_packages(),
    python_requires=">=3.8, <3.11",
    package_data={"tetra_model_zoo": ["**/*"]},\
    include_package_data=True,
    install_requires=[line.strip() for line in open(requirements_path).readlines()],
    extras_require={
        model_dir.name: [line.strip() for line in open(model_dir / r_file).readlines()]
        for model_dir in model_zoo_path.iterdir()
        # Collect all subdirs of model_zoo that have a requirements.txt file
        if not model_dir.is_file() and (model_dir / r_file).exists()
    },
    license="MIT",
)
