import numpy as np
import pytest
import torch

from tetra_model_zoo.openai_clip.app import ClipApp
from tetra_model_zoo.openai_clip.model import MODEL_ASSET_VERSION, MODEL_NAME, Clip
from tetra_model_zoo.utils.asset_loaders import load_image

IMAGE_ADDRESS = f"https://tetra-public-assets.s3.us-west-2.amazonaws.com/model-zoo/clip/v{MODEL_ASSET_VERSION}/demo_images/image1.jpg"

TEXT = "pyramid in desert"


@pytest.fixture(scope="module")
def source_clip_model() -> Clip:
    """Load model via OpenAI clip."""
    return Clip.from_pretrained()


@pytest.fixture(scope="module")
def clip_app(source_clip_model: Clip) -> ClipApp:
    # Load Application
    return ClipApp(source_clip_model)


@pytest.fixture(scope="module")
def processed_sample_image(clip_app: ClipApp) -> torch.Tensor:
    """Image preprocessing."""
    return clip_app.process_image(load_image(IMAGE_ADDRESS, MODEL_NAME))


@pytest.fixture(scope="module")
def processed_sample_text(clip_app: ClipApp) -> torch.Tensor:
    """Tokenzing text."""
    return clip_app.process_text(TEXT)


def test_prediction(
    clip_app: ClipApp,
    processed_sample_image: torch.Tensor,
    processed_sample_text: torch.Tensor,
):
    """Verify our driver produces the correct score given image and text pair."""
    assert clip_app.predict_similarity(processed_sample_image, processed_sample_text)


def test_numerical(
    source_clip_model: Clip,
    clip_app: ClipApp,
    processed_sample_image: torch.Tensor,
    processed_sample_text: torch.Tensor,
):
    """Verify that raw (numeric) outputs of both networks are the same."""
    source_clip_text_model, source_clip_image_model = (
        source_clip_model.text_encoder,
        source_clip_model.image_encoder,
    )
    text_features = source_clip_text_model(processed_sample_text)
    image_features = source_clip_image_model(processed_sample_image)
    source_out = image_features @ text_features.t()
    tetra_out = clip_app.predict_similarity(
        processed_sample_image, processed_sample_text
    )

    assert np.allclose(source_out.detach().numpy(), tetra_out)
