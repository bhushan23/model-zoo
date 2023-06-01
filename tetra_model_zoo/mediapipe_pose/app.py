from __future__ import annotations

from typing import List, Tuple

import torch

from tetra_model_zoo.mediapipe.app import MediaPipeApp
from tetra_model_zoo.mediapipe_pose.model import (
    DETECT_DSCALE,
    DETECT_DXY,
    DETECT_SCORE_SLIPPING_THRESHOLD,
    POSE_KEYPOINT_INDEX_END,
    POSE_KEYPOINT_INDEX_START,
    POSE_LANDMARK_CONNECTIONS,
    ROTATION_VECTOR_OFFSET_RADS,
    MediaPipePose,
)
from tetra_model_zoo.utils.bounding_box_processing import (
    compute_box_corners_with_rotation,
)
from tetra_model_zoo.utils.image_processing import compute_vector_rotation


class MediaPipePoseApp(MediaPipeApp):
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with MediaPipe's pose landmark detector.

    The app uses 2 models:
        * MediaPipePoseDetector
        * MediaPipePoseLandmark

    See the class comment for the parent class for details.
    """

    def __init__(
        self,
        model: MediaPipePose,
        min_detector_pose_box_score: float = 0.75,
        nms_iou_threshold: float = 0.3,
        min_landmark_score: float = 0.5,
    ):
        """
        Construct a mediapipe pose application.

        Inputs:
            model: MediaPipePose model
                Pose detection & landmark model container.

            See parent initializer for further parameter documentation.
        """

        def _landmark_detector_ignore_third_output(
            x: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            The Last landmark detector output ("mask") is not used by the demo application.
            Wrap the detector in a function that discards the mask.
            """
            out0, out1, _ = model.pose_landmark_detector(x)
            return out0, out1

        super().__init__(
            model.pose_detector,
            model.pose_detector_anchors,
            _landmark_detector_ignore_third_output,
            MediaPipePose.get_pose_detector_input_spec()["image"][0][-2:],
            MediaPipePose.get_pose_landmark_detector_input_spec(0)["image"][0][-2:],
            POSE_KEYPOINT_INDEX_START,
            POSE_KEYPOINT_INDEX_END,
            ROTATION_VECTOR_OFFSET_RADS,
            DETECT_DXY,
            DETECT_DSCALE,
            min_detector_pose_box_score,
            DETECT_SCORE_SLIPPING_THRESHOLD,
            nms_iou_threshold,
            min_landmark_score,
            POSE_LANDMARK_CONNECTIONS,
        )

    def _compute_object_roi(
        self,
        batched_selected_boxes: List[torch.Tensor | None],
        batched_selected_keypoints: List[torch.Tensor | None],
    ) -> List[torch.Tensor | None]:
        """
        See parent function for base functionality and parameter documentation.

        The MediaPipe pose pipeline computes the ROI not from the detector bounding box,
        but from specific detected keypoints. This override implements that behavior.
        """
        batched_selected_roi = []
        for boxes, keypoints in zip(batched_selected_boxes, batched_selected_keypoints):
            if boxes is None or keypoints is None:
                batched_selected_roi.append(None)
                continue

            # Compute bounding box center and rotation
            theta = compute_vector_rotation(
                keypoints[:, self.keypoint_rotation_vec_start_idx, ...],
                keypoints[:, self.keypoint_rotation_vec_end_idx, ...],
                self.rotation_offset_rads,
            )
            xc = keypoints[..., self.keypoint_rotation_vec_start_idx, 0]
            yc = keypoints[..., self.keypoint_rotation_vec_start_idx, 1]
            x1 = keypoints[..., self.keypoint_rotation_vec_end_idx, 0]
            y1 = keypoints[..., self.keypoint_rotation_vec_end_idx, 1]

            # Square box always
            w = ((xc - x1) ** 2 + (yc - y1) ** 2).sqrt() * 2 * self.detect_box_scale
            h = w

            # Compute box corners from box center, width, height
            batched_selected_roi.append(
                compute_box_corners_with_rotation(xc, yc, w, h, theta)
            )

        return batched_selected_roi
