import numpy as np

from tetra_model_zoo.litehrnet.app import LiteHRNetApp
from tetra_model_zoo.litehrnet.model import MODEL_ASSET_VERSION, MODEL_NAME, LiteHRNet
from tetra_model_zoo.utils.asset_loaders import MODEL_ZOO_ASSET_PATH, load_image

IMAGE_LOCAL_PATH = "litehrnet_demo.png"
IMAGE_ADDRESS = (
    f"{MODEL_ZOO_ASSET_PATH}/litehrnet/v{MODEL_ASSET_VERSION}/{IMAGE_LOCAL_PATH}"
)


EXPECTED_KEYPOINTS = np.array(
    [
        [
            [70, 34],
            [77, 32],
            [72, 30],
            [91, 37],
            [72, 32],
            [109, 67],
            [67, 67],
            [130, 104],
            [63, 104],
            [112, 125],
            [40, 102],
            [105, 144],
            [77, 144],
            [119, 202],
            [81, 190],
            [142, 251],
            [88, 230],
        ]
    ]
)


def test_numerical():
    image = load_image(IMAGE_ADDRESS, MODEL_NAME)
    litehrnet = LiteHRNet.from_pretrained()
    app = LiteHRNetApp(litehrnet, litehrnet.inferencer)
    keypoints = app.predict_pose_keypoints(image, True)

    np.testing.assert_allclose(
        np.asarray(EXPECTED_KEYPOINTS, dtype=np.float32),
        np.asarray(keypoints, dtype=np.float32),
        rtol=0.02,
        atol=1.5,
    )
