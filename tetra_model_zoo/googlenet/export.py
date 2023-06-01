from __future__ import annotations

import tetra_hub as hub
from tetra_model_zoo.googlenet.model import MODEL_NAME, GoogLeNet
from tetra_model_zoo.imagenet_classifier.model import trace_imagenet_classifier
from tetra_model_zoo.utils.args import base_export_parser
from tetra_model_zoo.utils.hub import download_hub_models


def main():
    # Export parameters
    parser = base_export_parser()

    args = parser.parse_args()

    # Instantiate the model
    model = GoogLeNet.from_pretrained()

    # Trace the model.
    traced_model = trace_imagenet_classifier(model)

    # Select the device(s) you'd like to optimize for.
    devices = [hub.Device(x) for x in args.devices]

    # Submit the traced models for conversion & profiling.
    jobs = hub.submit_profile_job(
        name=MODEL_NAME,
        model=traced_model,
        input_shapes=model.get_input_spec(),
        device=devices,
    )

    # Download the optimized asset!
    download_hub_models(jobs)


if __name__ == "__main__":
    main()
