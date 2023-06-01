import numpy as np

from tetra_model_zoo.real_esrgan.app import RealESRGANApp
from tetra_model_zoo.real_esrganv4.model import (
    MODEL_ASSET_VERSION,
    MODEL_NAME,
    RealESRGANv4,
)
from tetra_model_zoo.utils.asset_loaders import MODEL_ZOO_ASSET_PATH, load_image
from tetra_model_zoo.utils.testing import assert_most_same, skip_clone_repo_check

IMAGE_ADDRESS = (
    f"{MODEL_ZOO_ASSET_PATH}/realesrganx4/v{MODEL_ASSET_VERSION}/realesrganx4_demo.jpg"
)
OUTPUT_IMAGE_ADDRESS = f"{MODEL_ZOO_ASSET_PATH}/realesrganx4/v{MODEL_ASSET_VERSION}/realesrganx4_demo_output.png"


@skip_clone_repo_check
def test_numerical():
    image = load_image(IMAGE_ADDRESS, MODEL_NAME)
    model = RealESRGANv4.from_pretrained()
    app = RealESRGANApp(model=model)
    output_img = app.upscale_image(image)[0]

    expected_output_image = load_image(OUTPUT_IMAGE_ADDRESS, MODEL_NAME)
    assert_most_same(
        np.asarray(expected_output_image, dtype=np.float32),
        np.array(output_img).astype(np.float32),
        tolerance=0.01,
    )
