import argparse
import os

import numpy as np
import torch
from PIL import Image

from tetra_model_zoo.openai_clip.demo import DEFAULT_DEMO_IMAGE_FOLDER
from tetra_model_zoo.optimized_clip.app import OptimizedClipApp
from tetra_model_zoo.optimized_clip.model import MODEL_NAME, OptimizedClip
from tetra_model_zoo.utils.asset_loaders import MODEL_ZOO_STORE, load_image


# Run Clip on a directory of images with a query text.
# The demo will display similarity score for each image.
def main():
    # Demo parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir",
        type=str,
        default=DEFAULT_DEMO_IMAGE_FOLDER,
        help="Path to image directory",
    )
    parser.add_argument(
        "--image_names",
        type=str,
        default="image1.jpg,image2.jpg,image3.jpg",
        help="Specify names of the images in the folder.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="camping under the stars",
        help="Text prompt for image search",
    )
    args = parser.parse_args()

    # Load model
    clip_model = OptimizedClip.from_pretrained()
    app = OptimizedClipApp(clip=clip_model)

    image_names = args.image_names.split(",")
    text = app.process_text(args.text)
    images = []

    # Iterate through images and text provided by user
    for filename in image_names:
        # Make sure the file is an image
        if os.path.splitext(filename)[1].lower() in [".jpg", ".jpeg", ".png"]:

            # Preprocess image and text pair
            image = app.process_image(
                load_image(os.path.join(args.image_dir, filename), MODEL_NAME)
            )
            images.append(image)

        else:
            print(f"Skipping file {filename}")

    images = torch.stack(images).squeeze(1)

    # Compute similarity
    predictions = app.predict_similarity(images, text).flatten()

    # Display all the images and their score wrt to the text prompt provided.
    print(f"Searching images by prompt: {args.text}")
    for i in range(len(predictions)):
        print(
            f"\t Image with name: {image_names[i]} has a similarity score={predictions[i]}"
        )

    # Show image
    print("Displaying the most relevant image")
    most_relevant_image = Image.open(
        os.path.join(MODEL_ZOO_STORE, MODEL_NAME, image_names[np.argmax(predictions)])
    )
    most_relevant_image.show()


if __name__ == "__main__":
    main()
