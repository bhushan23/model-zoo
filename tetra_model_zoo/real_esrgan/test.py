import numpy as np

from tetra_model_zoo.real_esrgan.app import RealESRGANApp
from tetra_model_zoo.real_esrgan.model import (
    MODEL_ASSET_VERSION,
    MODEL_NAME,
    RealESRGAN,
)
from tetra_model_zoo.utils.asset_loaders import load_image
from tetra_model_zoo.utils.testing import skip_clone_repo_check

IMAGE_ADDRESS = f"https://tetra-public-assets.s3.us-west-2.amazonaws.com/model-zoo/realesrgan/v{MODEL_ASSET_VERSION}/realesrgan_demo.jpg"
OUTPUT_IMAGE_ADDRESS = f"https://tetra-public-assets.s3.us-west-2.amazonaws.com/model-zoo/realesrgan/v{MODEL_ASSET_VERSION}/realesrgan_demo_output.png"


@skip_clone_repo_check
def test_realesrgan_app():
    image = load_image(IMAGE_ADDRESS, MODEL_NAME)
    output_image = load_image(OUTPUT_IMAGE_ADDRESS, MODEL_NAME)
    model = RealESRGAN.from_pretrained()
    app = RealESRGANApp(model)
    app_output_image = app.upscale_image(image)[0]
    np.testing.assert_allclose(
        np.asarray(app_output_image, dtype=np.float32),
        np.asarray(output_image, dtype=np.float32),
        rtol=0.02,
        atol=1.5,
    )
