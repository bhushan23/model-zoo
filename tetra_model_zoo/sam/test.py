import numpy as np
import pytest
import torch

from tetra_model_zoo.sam import App
from tetra_model_zoo.sam.model import MODEL_NAME, SAMTetraWrapper
from tetra_model_zoo.utils.asset_loaders import load_image

IMAGE_ADDRESS = "https://raw.githubusercontent.com/tetraai/segment-anything/main/notebooks/images/truck.jpg"
# Test small model for quick turn-around.
TEST_MODEL_TYPE = "vit_b"


@pytest.fixture(scope="module")
def input_image_data() -> np.ndarray:
    return np.asarray(load_image(IMAGE_ADDRESS, MODEL_NAME))


@pytest.mark.skip(reason="Error: Process completed with exit code 143.")
def test_e2e_numerical(
    input_image_data: np.ndarray,
    monkeypatch,
):
    """Verify our driver produces the correct segmentation as source PyTorch model"""
    monkeypatch.setattr("builtins.input", lambda: "y")

    sam_wrapper = SAMTetraWrapper(TEST_MODEL_TYPE)
    sam_model = sam_wrapper.get_sam()
    sam_predictor = sam_wrapper.SamPredictor(sam_model)
    sam_decoder = sam_wrapper.SamOnnxModel(
        sam_model, orig_img_size=input_image_data.shape[:2], return_single_mask=True
    )

    sam_predictor.set_image(input_image_data)
    # Tetra SAM App for segmentation
    sam_app = App()
    # Prepare image for segmentation
    sam_app.prepare(input_image_data)

    # Ensure image embeddings match with source model
    np.allclose(
        sam_predictor.features.detach().numpy(),
        sam_app.image_embeddings.detach().numpy(),
    )

    #
    # Verify Decoder output is correct
    #

    # Create input for decoder
    embed_size = sam_predictor.model.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    decoder_inputs = {
        "image_embeddings": sam_predictor.features.detach(),
        "point_coords": torch.randint(low=0, high=500, size=(1, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1,), dtype=torch.float),
        "mask_input": torch.zeros(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
    }

    # Perform inference for decoder models
    obs_decoder_output = sam_app.generate_mask_from_points(
        decoder_inputs["point_coords"],
        decoder_inputs["point_labels"],
    )

    decoder_inputs["point_coords"] = decoder_inputs["point_coords"].unsqueeze(0)
    decoder_inputs["point_labels"] = decoder_inputs["point_labels"].unsqueeze(0)
    exp_decoder_output = sam_decoder(*decoder_inputs.values())

    # Ensure segmentation upscaled mask, scores and low-res masks match with source model
    for exp, obs in zip(exp_decoder_output, obs_decoder_output):
        np.allclose(exp.detach().numpy(), obs.detach().numpy())
