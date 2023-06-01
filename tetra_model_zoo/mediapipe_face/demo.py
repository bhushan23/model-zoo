import argparse

import numpy as np
from PIL import Image

from tetra_model_zoo.mediapipe_face.app import MediaPipeFaceApp
from tetra_model_zoo.mediapipe_face.model import MODEL_NAME, MediaPipeFace
from tetra_model_zoo.utils.asset_loaders import load_image
from tetra_model_zoo.utils.camera_capture import capture_and_display_processed_frames


#
# Run Mediapipe Face landmark detection end-to-end on a sample image or camera stream.
# The demo will display output with the predicted landmarks & bounding boxes drawn.
#
def main():
    # Demo parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        required=False,
        help="image file path or URL",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera Input ID",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.75,
        help="Score threshold for NonMaximumSuppression",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.3,
        help="Intersection over Union (IoU) threshold for NonMaximumSuppression",
    )

    args = parser.parse_args()

    # Load app
    app = MediaPipeFaceApp(
        MediaPipeFace.from_pretrained(),
        args.score_threshold,
        args.iou_threshold,
    )
    print("Model and App Loaded")

    if args.image:
        image = load_image(args.image, MODEL_NAME).convert("RGB")
        pred_image = app.predict_landmarks_from_image(image)
        Image.fromarray(pred_image[0], "RGB").show()
    else:

        def frame_processor(frame: np.ndarray) -> np.ndarray:
            return app.predict_landmarks_from_image(frame)[0]  # type: ignore

        capture_and_display_processed_frames(
            frame_processor, "Tetra Mediapipe Face Demo", args.camera
        )


if __name__ == "__main__":
    main()
