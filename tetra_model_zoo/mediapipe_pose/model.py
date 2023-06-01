from __future__ import annotations

from typing import Callable, Tuple

import torch

from tetra_model_zoo.mediapipe.utils import MediaPipePyTorchAsRoot
from tetra_model_zoo.utils.input_spec import InputSpec

MODEL_NAME = "mediapipe_pose"

POSE_LANDMARK_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 13),
    (13, 15),
    (15, 17),
    (17, 19),
    (19, 15),
    (15, 21),
    (12, 14),
    (14, 16),
    (16, 18),
    (18, 20),
    (20, 16),
    (16, 22),
    (11, 12),
    (12, 24),
    (24, 23),
    (23, 11),
]


# pose detector model parameters.
BATCH_SIZE = 1
DETECT_SCORE_SLIPPING_THRESHOLD = 100  # Clip output scores to this maximum value.
DETECT_DXY, DETECT_DSCALE = (
    0,
    1.5,
)  # Modifiers applied to pose detector output bounding box to encapsulate the entire pose.
POSE_KEYPOINT_INDEX_START = 2  # The pose detector outputs several keypoints. This is the keypoint index for the bottom.
POSE_KEYPOINT_INDEX_END = 3  # The pose detector outputs several keypoints. This is the keypoint index for the top.
ROTATION_VECTOR_OFFSET_RADS = (
    torch.pi / 2
)  # Offset required when computing rotation of the detected pose.


class MediaPipePose:
    def __init__(
        self,
        pose_detector: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        pose_detector_anchors: torch.Tensor,
        pose_landmark_detector: Callable[
            [torch.Tensor],
            Tuple[
                torch.Tensor,
                torch.Tensor,
            ],
        ],
    ) -> None:
        """
        Construct a mediapipe pose model.

        Inputs:
            pose_detector: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
                Pose detection model. Input is an image, output is
                [bounding boxes & keypoints, box & kp scores]

            pose_detector_anchors: torch.Tensor
                Anchor boxes used to decode the output boxes & keypoints from the pose detector model.

            pose_landmark_detector
                Face landmark detector model. Input is an image cropped to the posing object. The pose must be upright
                and un-tilted in the frame. Returns [landmark_scores, landmarks, mask]

                Note that although the landmark detector returns 3 values,
                the third output (mask) is unused by this application.

        """
        super().__init__()
        self.pose_detector = pose_detector
        self.pose_detector_anchors = pose_detector_anchors
        self.pose_landmark_detector = pose_landmark_detector

    @staticmethod
    def from_pretrained(
        detector_weights: str = "blazepose.pth",
        detector_anchors: str = "anchors_pose.npy",
        landmark_detector_weights: str = "blazepose_landmark.pth",
    ) -> MediaPipePose:
        return MediaPipePose(
            *MediaPipePose._load_mediapipe_pose_models(
                detector_weights, detector_anchors, landmark_detector_weights
            )
        )

    @staticmethod
    def get_pose_detector_input_spec(batch_size: int = BATCH_SIZE) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type) of the pose detector.
        This can be used to submit profiling job on TetraHub.
        """
        return {"image": ((batch_size, 3, 128, 128), "float32")}

    @staticmethod
    def get_pose_landmark_detector_input_spec(batch_size: int) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type) of the pose landmark detector.
        This can be used to submit profiling job on TetraHub.
        """
        return {"image": ((batch_size, 3, 256, 256), "float32")}

    @staticmethod
    def _load_mediapipe_pose_models(
        detector_weights: str, detector_anchors: str, landmark_detector_weights: str
    ) -> Tuple[torch.nn.Module, torch.Tensor, torch.nn.Module]:
        """
        Load mediapipe models from the source repository.
        Returns tuple[<source repository>.blazepose.BlazePose, BlazePose Anchors, <source repository>.blazepose_landmark.BlazePoseLandmark]
        """
        with MediaPipePyTorchAsRoot():
            from blazepose import BlazePose
            from blazepose_landmark import BlazePoseLandmark

            pose_detector = BlazePose()
            pose_detector.load_weights(detector_weights)
            pose_detector.load_anchors(detector_anchors)
            pose_regressor = BlazePoseLandmark()
            pose_regressor.load_weights(landmark_detector_weights)

            return (pose_detector, pose_detector.anchors, pose_regressor)
