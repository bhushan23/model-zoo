import numpy as np

from tetra_model_zoo.esr_gan.app import ESRGANApp
from tetra_model_zoo.esr_gan.model import ESRGAN, MODEL_ASSET_VERSION, MODEL_NAME
from tetra_model_zoo.utils.asset_loaders import load_image
from tetra_model_zoo.utils.testing import skip_clone_repo_check

IMAGE_ADDRESS = f"https://tetra-public-assets.s3.us-west-2.amazonaws.com/model-zoo/esrgan/v{MODEL_ASSET_VERSION}/esrgan_demo.jpg"
OUTPUT_IMAGE_ADDRESS = f"https://tetra-public-assets.s3.us-west-2.amazonaws.com/model-zoo/esrgan/v{MODEL_ASSET_VERSION}/esrgan_demo_output.png"


@skip_clone_repo_check
def test_esrgan_app():
    image = load_image(IMAGE_ADDRESS, MODEL_NAME)
    output_image = load_image(OUTPUT_IMAGE_ADDRESS, MODEL_NAME)
    app = ESRGANApp(ESRGAN.from_pretrained())
    app_output_image = app.upscale_image(image)
    np.testing.assert_allclose(
        np.asarray(app_output_image, dtype=np.float32) / 255,
        np.asarray(output_image, dtype=np.float32) / 255,
        rtol=0.02,
        atol=0.2,
    )
