from __future__ import annotations

from typing import Any, Tuple

import torch

import tetra_hub as hub
from tetra_model_zoo.mediapipe.utils import trace_mediapipe
from tetra_model_zoo.mediapipe_face.model import MediaPipeFace
from tetra_model_zoo.utils.args import base_export_parser
from tetra_model_zoo.utils.hub import download_hub_models


def trace(
    face_detector: torch.nn.Module, landmark_detector: torch.nn.Module
) -> Tuple[Any, Any]:
    """
    Convert the models to pytorch traces. Traces can be saved & loaded from disk.
    With Tetra Hub, a pytorch trace can be exported to run efficiently on mobile devices!

    Inputs:
        face_detector
            Face detetctor model, from load_mediapipe_face_models()

        face_landmark_detector
            Face landmark detector model, from load_mediapipe_face_models()

    Returns: Tuple[Face Detector Trace Object, Landmark Detector Trace Object]
    """
    return trace_mediapipe(
        MediaPipeFace.get_face_detector_input_spec(),
        face_detector,
        MediaPipeFace.get_face_landmark_detector_input_spec(1),
        landmark_detector,
    )


def main():
    # Export parameters
    parser = base_export_parser()
    args = parser.parse_args()

    # Load source models
    model = MediaPipeFace.from_pretrained()
    face_detector = model.face_detector
    face_landmark = model.face_landmark_detector

    # Trace the model.
    traced_encoder, traced_decoder = trace(face_detector, face_landmark)  # type: ignore

    # Select the device you'd like to optimize for.
    devices = [hub.Device(device) for device in args.devices]

    # Submit the traced models for conversion & profiling.
    box_detector_jobs = hub.submit_profile_job(
        name="face_detector",
        model=traced_encoder,
        input_shapes=MediaPipeFace.get_face_detector_input_spec(),
        device=devices,
    )

    landmark_detector_jobs = hub.submit_profile_job(
        name="face_landmark_detector",
        model=traced_decoder,
        input_shapes=MediaPipeFace.get_face_landmark_detector_input_spec(1),
        device=devices,
    )

    # Download the optimized assets!
    print("Exported face detector(s):\n")
    download_hub_models(box_detector_jobs)
    print("Exported face landmark detector(s):\n")
    download_hub_models(landmark_detector_jobs)


if __name__ == "__main__":
    main()
