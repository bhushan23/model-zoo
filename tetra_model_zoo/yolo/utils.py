from __future__ import annotations

import torch


def transform_box_layout_xywh2xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes with (xywh) layout to (xyxy)

    Parameters:
        boxes (torch.Tensor): Input boxes with layout (xywh)

    Returns:
        torch.Tensor: Output box with layout (xyxy)
            i.e. [top_left_x | top_left_y | bot_right_x | bot_right_y]
    """
    # Convert to (x1, y1, x2, y2)
    top_left_x = boxes[..., 0] - boxes[..., 2] / 2
    top_left_y = boxes[..., 1] - boxes[..., 3] / 2
    bot_right_x = boxes[..., 0] + boxes[..., 2] / 2
    bot_right_y = boxes[..., 1] + boxes[..., 3] / 2
    boxes = torch.stack((top_left_x, top_left_y, bot_right_x, bot_right_y), -1)
    return boxes


def detect_postprocess(detector_output: torch.Tensor):
    """
    Post processing to break Yolo(v6,v7) detector output into multiple, consumable tensors (eg. for NMS).
        such as bounding boxes, classes, and confidence.

    Parameters:
        detector_output: torch.Tensor
            The output of Yolo Detection model
            Shape is [batch, num_preds, k]
                where, k = # of classes + 5
                k is structured as follows [boxes (4) : conf (1) : # of classes]
                and boxes are co-ordinates [x_center, y_center, w, h]

    Returns:
        boxes: torch.Tensor
            Bounding box locations. Shape is [batch, num preds, 4] where 4 == (x1, y1, x2, y2)
        scores: torch.Tensor
            class scores multiplied by confidence: Shape is [batch, num_preds]
        class_idx: torch.tensor
            Shape is [batch, num_preds, 1] where the last dim is the index of the most probable class of the prediction.
    """
    # Break output into parts
    boxes = detector_output[:, :, :4]
    conf = detector_output[:, :, 4:5]
    scores = detector_output[:, :, 5:]

    # Convert boxes to (x1, y1, x2, y2)
    boxes = transform_box_layout_xywh2xyxy(boxes)

    # Combine confidence and scores.
    scores *= conf

    # Get class ID of most likely score.
    scores, class_idx = get_most_likely_score(scores)

    return boxes, scores, class_idx


def get_most_likely_score(scores: torch.Tensor):
    """
    Returns most likely score and class id

    Args:
        scores (torch.tensor): final score after post-processing predictions

    Returns:
        scores: torch.Tensor
            class scores reduced to keep max score per prediction
            Shape is [batch, num_preds]
        class_idx: torch.tensor
            Shape is [batch, num_preds] where the last dim is the index of the most probable class of the prediction.
    """
    scores, class_idx = torch.max(scores, -1, keepdim=False)
    return scores, class_idx.float()
