import clip
import numpy as np
import pytest
import torch

from tetra_model_zoo.openai_clip.model import MODEL_ASSET_VERSION, load_clip
from tetra_model_zoo.optimized_clip import transformer_utils
from tetra_model_zoo.optimized_clip.app import OptimizedClipApp
from tetra_model_zoo.optimized_clip.model import MODEL_NAME, OptimizedClip
from tetra_model_zoo.utils.asset_loaders import callback_with_retry, load_image
from tetra_model_zoo.utils.compare import compare_psnr

PSNR_THRESHOLD = 40
IMAGE_ADDRESS = f"https://tetra-public-assets.s3.us-west-2.amazonaws.com/model-zoo/clip/v{MODEL_ASSET_VERSION}/demo_images/image1.jpg"
TEXT = "pyramid in desert"


# Utilities
def compare_output_to_ane_output(
    output_: torch.Tensor,
    ane_output_: torch.Tensor,
    psnr_threshold: int = PSNR_THRESHOLD,
    permute_output: bool = True,
) -> None:
    if permute_output:
        ane_output_ = ane_output_.squeeze(0).permute([2, 1, 0])
    assert output_.shape == ane_output_.shape
    compare_psnr(output_, ane_output_, psnr_threshold)


# Fixtures
@pytest.fixture
def clip_vision_transformers():
    # Load Clip: Create ANE clip from it
    model, preprocess = callback_with_retry(num_retries=5, callback=load_clip)
    vision_transformer = model.visual
    ane_vision_transformer = transformer_utils.VisionTransformer.from_clip(model)

    # return both
    return vision_transformer, ane_vision_transformer


@pytest.fixture
def clip_transformers():
    model, preprocess = clip.load("ViT-B/16", device="cpu")

    attention_mask = torch.empty(77, 77)
    attention_mask.fill_(float("-inf"))
    attention_mask.triu_(1)  # zero out the lower diagonal
    attention_mask = attention_mask.permute(1, 0).unsqueeze(0).unsqueeze(-2)
    transformer = model.transformer
    width = transformer.width
    layers = transformer.layers

    # Create ane ANE transformer
    ane_transformer = transformer_utils.Transformer(width, layers, 8, attention_mask)
    ane_transformer.load_state_dict(transformer.state_dict())

    # return both
    return transformer, ane_transformer


@pytest.fixture
def input_data():
    width, grid = 512, 77
    input_ = torch.rand(grid, 1, width)
    ane_input_ = input_.permute([2, 1, 0]).unsqueeze(0)
    return input_, ane_input_


@pytest.fixture(scope="module")
def source_clip_model() -> OptimizedClip:
    """Load text and image encoder for Optimized clip."""
    return OptimizedClip.from_pretrained()


@pytest.fixture(scope="module")
def clip_app(source_clip_model: OptimizedClip) -> OptimizedClipApp:
    # Load Application
    return OptimizedClipApp(source_clip_model)


@pytest.fixture(scope="module")
def processed_sample_image(clip_app: OptimizedClipApp) -> torch.Tensor:
    """Image preprocessing."""
    return clip_app.process_image(load_image(IMAGE_ADDRESS, MODEL_NAME))


@pytest.fixture(scope="module")
def processed_sample_text(clip_app: OptimizedClipApp) -> torch.Tensor:
    """Tokenzing text."""
    return clip_app.process_text(TEXT)


# Unit tests
def test_ane_layer_norm(clip_transformers, input_data):
    # Load models
    transformer, ane_transformer = clip_transformers
    input_, ane_input_ = input_data

    for i in range(len(transformer.resblocks)):
        # Load blocks
        layer_norm = transformer.resblocks[i].ln_1
        ane_layer_norm = ane_transformer.resblocks[i].ln_1

        # Run models
        output_ = layer_norm(input_)
        ane_output_ = ane_layer_norm(ane_input_)

        # Compare results
        compare_output_to_ane_output(output_, ane_output_, 90)


def test_mlp(clip_transformers, input_data):
    # Load models
    transformer, ane_transformer = clip_transformers
    input_, ane_input_ = input_data

    for i in range(len(transformer.resblocks)):
        # Load blocks
        mlp = transformer.resblocks[i].mlp
        ane_mlp = ane_transformer.resblocks[i].mlp

        # Run models
        output_ = mlp(input_)
        ane_output_ = ane_mlp(ane_input_)

        # Compare results
        compare_output_to_ane_output(output_, ane_output_, 90)


def test_attention(clip_transformers, input_data):
    # Load models
    transformer, ane_transformer = clip_transformers
    input_, ane_input_ = input_data

    for i in range(len(transformer.resblocks)):
        # Load blocks
        attn = transformer.resblocks[i].attn
        ane_attn = ane_transformer.resblocks[i].attn

        # Run models
        output_, _ = attn(input_, input_, input_)
        ane_output_, _ = ane_attn(ane_input_, ane_input_, ane_input_)

        # Compare results
        compare_output_to_ane_output(output_, ane_output_, 90)


def test_resblock(clip_transformers, input_data):
    # Load models
    transformer, ane_transformer = clip_transformers
    input_, ane_input_ = input_data

    for i in range(len(transformer.resblocks)):
        # Load blocks
        resblock = transformer.resblocks[i]
        ane_resblock = ane_transformer.resblocks[i]

        # Run models
        output_ = resblock(input_)
        ane_output_ = ane_resblock(ane_input_)

        # Compare results
        compare_output_to_ane_output(output_, ane_output_, 90)


def test_mha(clip_transformers, input_data):
    # Load models
    transformer, ane_transformer = clip_transformers
    input_, ane_input_ = input_data

    for i in range(len(transformer.resblocks)):
        # Load blocks
        mha = transformer.resblocks[i].attn
        ane_mha = ane_transformer.resblocks[i].attn

        # Run models
        output_ = mha(query=input_, key=input_, value=input_)
        ane_output_ = ane_mha(ane_input_, ane_input_, ane_input_)

        # Compare results
        compare_output_to_ane_output(output_[0], ane_output_[0], 90)


def test_transformer(clip_transformers, input_data):
    # Load models
    transformer, ane_transformer = clip_transformers
    input_, ane_input_ = input_data

    # Run models
    output_ = transformer(input_)
    ane_output_ = ane_transformer(ane_input_)

    # Compare results
    compare_output_to_ane_output(output_, ane_output_, 90)


def test_ane_vision_transformer(clip_vision_transformers):
    vision_transformer, ane_vision_transformer = clip_vision_transformers

    input_ = 120 * torch.rand(1, 3, 224, 224)
    output_ = vision_transformer(input_)
    ane_output_ = ane_vision_transformer(input_)
    ane_output_ = ane_output_.squeeze(-1).squeeze(-1)

    # Compare results
    compare_output_to_ane_output(output_, ane_output_, 90, False)


def test_prediction(
    clip_app: OptimizedClipApp,
    processed_sample_image: torch.Tensor,
    processed_sample_text: torch.Tensor,
):
    """Verify our driver produces the correct score given image and text pair."""
    assert clip_app.predict_similarity(processed_sample_image, processed_sample_text)


# End to End tests
def test_numerical(
    source_clip_model: OptimizedClip,
    clip_app: OptimizedClipApp,
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
    source_out = image_features @ text_features
    tetra_out = clip_app.predict_similarity(
        processed_sample_image, processed_sample_text
    )

    # Compare
    assert np.allclose(source_out.detach().numpy(), tetra_out)
