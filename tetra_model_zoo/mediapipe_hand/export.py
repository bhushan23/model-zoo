from __future__ import annotations

from typing import Any, Tuple

import torch

import tetra_hub as hub
from tetra_model_zoo.mediapipe.utils import trace_mediapipe
from tetra_model_zoo.mediapipe_hand.model import MediaPipeHand
from tetra_model_zoo.utils.args import base_export_parser
from tetra_model_zoo.utils.hub import download_hub_models


def trace(
    palm_detector: torch.nn.Module, hand_landmark_detector: torch.nn.Module
) -> Tuple[Any, Any]:
    """
    Convert the models to pytorch traces. Traces can be saved & loaded from disk.
    With Tetra Hub, a pytorch trace can be exported to run efficiently on mobile devices!

    Inputs:
        palm_detector
            Palm detetctor model, from load_mediapipe_hand_models()

        hand_landmark_detector
            Hand landmark detector model, from load_mediapipe_hand_models()

    Returns: Tuple[Hand Detector Trace Object, Landmark Detector Trace Object]
    """
    return trace_mediapipe(
        MediaPipeHand.get_palm_detector_input_spec(),
        palm_detector,
        MediaPipeHand.get_hand_landmark_detector_input_spec(1),
        hand_landmark_detector,
    )


def main():
    # Export parameters
    parser = base_export_parser()
    args = parser.parse_args()

    # Load source models
    model = MediaPipeHand.from_pretrained()

    hand_detector = model.hand_detector
    hand_landmark = model.hand_landmark_detector

    # Trace the model.
    traced_encoder, traced_decoder = trace(hand_detector, hand_landmark)  # type: ignore

    # Select the device you'd like to optimize for.
    devices = [hub.Device(device) for device in args.devices]

    # Submit the traced models for conversion & profiling.
    box_detector_jobs = hub.submit_profile_job(
        name="hand_palm_detector",
        model=traced_encoder,
        input_shapes=MediaPipeHand.get_palm_detector_input_spec(),
        device=devices,
    )

    landmark_detector_jobs = hub.submit_profile_job(
        name="hand_landmark_detector",
        model=traced_decoder,
        input_shapes=MediaPipeHand.get_hand_landmark_detector_input_spec(1),
        device=devices,
    )

    # Download the optimized assets!
    print("Exported hand detector(s):\n")
    download_hub_models(box_detector_jobs)
    print("Exported hand landmark detector(s):\n")
    download_hub_models(landmark_detector_jobs)


if __name__ == "__main__":
    main()
