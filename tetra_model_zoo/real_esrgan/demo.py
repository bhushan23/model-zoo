import argparse

from tetra_model_zoo.real_esrgan.app import RealESRGANApp
from tetra_model_zoo.real_esrgan.model import (
    MODEL_ASSET_VERSION,
    MODEL_NAME,
    RealESRGAN,
)
from tetra_model_zoo.utils.asset_loaders import load_image

WEIGHTS_HELP_MSG = "RealESRGAN checkpoint `.pth` name from the Real-ESRGAN repo. Can be set to any of the model names defined here: https://github.com/xinntao/Real-ESRGAN/blob/master/docs/model_zoo.md to automatically download the file instead."


#
# Run Real-ESRGAN end-to-end on a sample image.
# The demo will display a image with the predicted bounding boxes.
#
def main():
    # Demo parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=str, default="realesr-general-x4v3", help=WEIGHTS_HELP_MSG
    )
    parser.add_argument(
        "--image",
        type=str,
        default=f"https://tetra-public-assets.s3.us-west-2.amazonaws.com/model-zoo/realesrgan/v{MODEL_ASSET_VERSION}/realesrgan_demo.jpg",
        help="image file path or URL.",
    )

    args = parser.parse_args()

    # Load image & model
    model = RealESRGAN.from_pretrained(args.weights)
    app = RealESRGANApp(model)
    print("Model Loaded")
    image = load_image(args.image, MODEL_NAME)
    pred_images = app.upscale_image(image)
    image.show()
    pred_images[0].show()


if __name__ == "__main__":
    main()
