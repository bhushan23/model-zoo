import os
import pathlib

import yaml
from schema import And, Optional, Schema, SchemaError

# Schema for info.yaml
# Ref: https://tetraai.sharepoint.com/:w:/g/EbEyQ34FFFlAteeJaDQu1oUBYqLaezu_7RO6StHJCYAAvg?e=1harYb
schema = Schema(
    {
        "name": And(str),
        "id": And(str),
        "headline": And(str),
        "domain": And(str),
        "use_case": And(str),
        "tags": And(lambda s: len(s) >= 0),
        "research_paper": And(str),
        "source_repo": And(str),
        "tetra_repo": And(str),
        "example_use": Optional(str),
        "technical_details": And(dict),
        "applicable_scenarios": And(lambda s: len(s) >= 0),
        "related_models": And(lambda s: len(s) >= 0),
        "form_factors": And(lambda s: len(s) >= 0),
        "static_image": And(str),
        "animated_asset": And(str),
    }
)

VALID_FORM_FACTORS = {"Phone", "Tablet", "Desktop", "IoT", "Automotive"}
VALID_DOMAINS = {"Computer Vision", "Audio", "Multimodal"}
VALID_USE_CASES = {
    "Image Classification",
    "Image Editing",
    "Super Resolution",
    "Segmentation",
    "Semantic Segmentation",
    "Video Classification",
    "Pose Detection",
    "Object Detection",
    "Pose Estimation",
    "Image To Text",
    "Speech Recognition",
}


def test_info_spec():
    # model-zoo models dir path
    tetra_model_zoo_path = pathlib.Path(__file__).parent.parent

    # collect all the models with info.yaml
    model_dir_with_spec = []
    for model_dir in os.listdir(tetra_model_zoo_path):
        info_spec = os.path.join(tetra_model_zoo_path, model_dir, "info.yaml")
        if os.path.exists(info_spec) and os.path.isfile(info_spec):
            model_dir_with_spec.append(model_dir)

    # validate info.yaml spec
    for model_dir in model_dir_with_spec:
        info_spec = os.path.join(tetra_model_zoo_path, model_dir, "info.yaml")
        with open(info_spec) as f:
            data = yaml.safe_load(f)

            try:
                # Validate high level-schema
                schema.validate(data)
            except SchemaError as e:
                assert 0, f"{e.code} in {info_spec}"

            #
            # Validate fields has correct values
            #

            # Validate model zoo domains
            domain = data["domain"]
            assert domain in VALID_DOMAINS

            use_case = data["use_case"]
            assert use_case in VALID_USE_CASES

            # Validate model zoo id is same as model dir
            assert model_dir == data["id"]

            # Validate related models are present
            related_models = data["related_models"]
            for r_model in related_models:
                assert r_model in model_dir_with_spec

            form_factors = data["form_factors"]
            for form_factor in form_factors:
                assert form_factor in VALID_FORM_FACTORS
