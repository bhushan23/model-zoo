import argparse

from PIL import Image

from tetra_model_zoo.utils.asset_loaders import load_image
from tetra_model_zoo.yolov7.app import YoloV7App
from tetra_model_zoo.yolov7.model import (
    DEFAULT_WEIGHTS,
    MODEL_ASSET_VERSION,
    MODEL_NAME,
    YoloV7,
)

WEIGHTS_HELP_MSG = "YoloV7 checkpoint `.pt` path on disk. Can be set to any of the strings defined here: https://github.com/WongKinYiu/yolov7/blob/main/utils/google_utils.py#L29 to automatically download the file instead."


#
# Run YoloV7 end-to-end on a sample image.
# The demo will display a image with the predicted bounding boxes.
#
def main():
    # Demo parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=str, default=DEFAULT_WEIGHTS, help=WEIGHTS_HELP_MSG
    )
    parser.add_argument(
        "--image",
        type=str,
        default=f"https://tetra-public-assets.s3.us-west-2.amazonaws.com/model-zoo/yolov7/v{MODEL_ASSET_VERSION}/yolov7_demo_640.jpg",
        help=f"image file path or URL. Image spatial dimensions (x and y) must be multiples of {YoloV7.STRIDE_MULTIPLE}",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.45,
        help="Score threshold for NonMaximumSuppression",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.7,
        help="Intersection over Union (IoU) threshold for NonMaximumSuppression",
    )

    args = parser.parse_args()

    # Load image & model
    model = YoloV7.from_pretrained(args.weights)
    app = YoloV7App(model, args.score_threshold, args.iou_threshold)
    print("Model Loaded")
    image = load_image(args.image, MODEL_NAME)
    pred_images = app.predict_boxes_from_image(image)
    Image.fromarray(pred_images[0]).show()


if __name__ == "__main__":
    main()
