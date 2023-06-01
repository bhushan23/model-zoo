from __future__ import annotations

from tetra_model_zoo.mediapipe.app import MediaPipeApp
from tetra_model_zoo.mediapipe_face.model import (
    DETECT_DSCALE,
    DETECT_DXY,
    DETECT_SCORE_SLIPPING_THRESHOLD,
    FACE_LANDMARK_CONNECTIONS,
    LEFT_EYE_KEYPOINT_INDEX,
    RIGHT_EYE_KEYPOINT_INDEX,
    ROTATION_VECTOR_OFFSET_RADS,
    MediaPipeFace,
)


class MediaPipeFaceApp(MediaPipeApp):
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with MediaPipe's hand landmark detector.

    The app uses 2 models:
        * MediaPipeFaceDetector
        * MediaPipeFaceLandmark

    See the class comment for the parent class for details.
    """

    def __init__(
        self,
        model: MediaPipeFace,
        min_detector_face_box_score: float = 0.75,
        nms_iou_threshold: float = 0.3,
        min_landmark_score: float = 0.5,
    ):
        """
        Construct a mediapipe face application.

        Inputs:
            model: MediaPipeFace model
                Face detection & landmark model container.

            See parent initializer for further parameter documentation.
        """
        super().__init__(
            model.face_detector,
            model.face_detector_anchors,
            model.face_landmark_detector,
            MediaPipeFace.get_face_detector_input_spec()["image"][0][-2:],
            MediaPipeFace.get_face_landmark_detector_input_spec(0)["image"][0][-2:],
            RIGHT_EYE_KEYPOINT_INDEX,
            LEFT_EYE_KEYPOINT_INDEX,
            ROTATION_VECTOR_OFFSET_RADS,
            DETECT_DXY,
            DETECT_DSCALE,
            min_detector_face_box_score,
            DETECT_SCORE_SLIPPING_THRESHOLD,
            nms_iou_threshold,
            min_landmark_score,
            FACE_LANDMARK_CONNECTIONS,
        )
