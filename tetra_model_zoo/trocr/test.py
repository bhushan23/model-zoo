import numpy as np
import pytest
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from tetra_model_zoo.trocr.app import TrOCRApp
from tetra_model_zoo.trocr.demo import DEFAULT_SAMPLE_IMAGE
from tetra_model_zoo.trocr.model import HUGGINGFACE_TROCR_MODEL, MODEL_NAME, TrOCR
from tetra_model_zoo.utils.asset_loaders import load_image

IMAGE_TEXT = 'industrial " Mr. Brown commented icity., letus have a'


@pytest.fixture(scope="module")
def source_huggingface_model() -> VisionEncoderDecoderModel:
    return VisionEncoderDecoderModel.from_pretrained(
        HUGGINGFACE_TROCR_MODEL, return_dict=False
    )  # type: ignore


@pytest.fixture(scope="module")
def trocr_app(source_huggingface_model: VisionEncoderDecoderModel) -> TrOCRApp:
    # Load Huggingface source
    source_model = source_huggingface_model
    io_processor = TrOCRProcessor.from_pretrained(HUGGINGFACE_TROCR_MODEL)

    # Load Application
    return TrOCRApp(TrOCR.from_source_model(source_model, io_processor))


@pytest.fixture(scope="module")
def processed_sample_image(trocr_app: TrOCRApp) -> torch.Tensor:
    """Huggingface-provided image preprocessing and token decoding."""
    return trocr_app.preprocess_image(load_image(DEFAULT_SAMPLE_IMAGE, MODEL_NAME))


def test_predict_text_from_image(
    trocr_app: TrOCRApp, processed_sample_image: torch.Tensor
):
    """Verify our driver produces the correct sentences from a given image input."""
    assert trocr_app.predict_text_from_image(processed_sample_image)[0] == IMAGE_TEXT


def test_numerical(
    source_huggingface_model: VisionEncoderDecoderModel,
    trocr_app: TrOCRApp,
    processed_sample_image: torch.Tensor,
):
    """Verify that raw (numeric) outputs of both networks are the same."""
    source_out = source_huggingface_model.generate(processed_sample_image).numpy()
    tetra_out = trocr_app.predict_text_from_image(
        processed_sample_image, raw_output=True
    )

    assert np.allclose(source_out, tetra_out)
