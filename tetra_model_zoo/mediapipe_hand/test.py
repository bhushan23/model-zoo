import numpy as np

from tetra_model_zoo.mediapipe_hand.app import MediaPipeHandApp
from tetra_model_zoo.mediapipe_hand.model import MODEL_NAME, MediaPipeHand
from tetra_model_zoo.utils.asset_loaders import MODEL_ZOO_ASSET_PATH, load_image
from tetra_model_zoo.utils.testing import skip_clone_repo_check

INPUT_IMAGE_ADDRESS = f"{MODEL_ZOO_ASSET_PATH}/mediapipe/v1/hand.jpeg"
OUTPUT_IMAGE_ADDRESS = f"{MODEL_ZOO_ASSET_PATH}/mediapipe/v1/hand_output.png"
# Because we have not made a modification to the pytorch source network,
# no numerical tests are included for the model; only for the app.


@skip_clone_repo_check
def test_hand_app():
    input = load_image(
        INPUT_IMAGE_ADDRESS,
        MODEL_NAME,
    )
    expected_output = load_image(
        OUTPUT_IMAGE_ADDRESS,
        MODEL_NAME,
    ).convert("RGB")
    app = MediaPipeHandApp(MediaPipeHand.from_pretrained())
    assert np.allclose(
        app.predict_landmarks_from_image(input)[0], np.asarray(expected_output)
    )
