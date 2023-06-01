import numpy as np

from tetra_model_zoo.mediapipe_pose.app import MediaPipePoseApp
from tetra_model_zoo.mediapipe_pose.model import MODEL_NAME, MediaPipePose
from tetra_model_zoo.utils.asset_loaders import MODEL_ZOO_ASSET_PATH, load_image
from tetra_model_zoo.utils.testing import skip_clone_repo_check

INPUT_IMAGE_ADDRESS = f"{MODEL_ZOO_ASSET_PATH}/mediapipe/v1/pose.jpeg"
OUTPUT_IMAGE_ADDRESS = f"{MODEL_ZOO_ASSET_PATH}/mediapipe/v1/pose_output.png"
# Because we have not made a modification to the pytorch source network,
# no numerical tests are included for the model; only for the app.


@skip_clone_repo_check
def test_pose_app():
    input = load_image(
        INPUT_IMAGE_ADDRESS,
        MODEL_NAME,
    )
    expected_output = load_image(
        OUTPUT_IMAGE_ADDRESS,
        MODEL_NAME,
    ).convert("RGB")
    app = MediaPipePoseApp(MediaPipePose.from_pretrained())
    assert np.allclose(
        app.predict_landmarks_from_image(input)[0], np.asarray(expected_output)
    )
