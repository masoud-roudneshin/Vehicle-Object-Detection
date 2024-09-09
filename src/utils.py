
import torch
import torch.nn as nn

def bbox_ciou(pred_boxes, target_boxes):
    """
    Calculate the CIoU (Complete IoU) loss between predicted boxes and target boxes.
    Arguments:
    - pred_boxes: [batch_size, 4] (x_center, y_center, width, height)
    - target_boxes: [batch_size, 4] (x_center, y_center, width, height)
    Returns:
    - ciou_loss: scalar loss value
    """
    pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

    target_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
    target_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
    target_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
    target_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

    # Intersection
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)

    # Union
    union_area = pred_area + target_area - inter_area
    IoU = inter_area / union_area

    # Calculate distance between center points
    center_dist = (pred_boxes[:, 0] - target_boxes[:, 0]) ** 2 + (pred_boxes[:, 1] - target_boxes[:, 1]) ** 2

    # Enclosing box
    enclosing_x1 = torch.min(pred_x1, target_x1)
    enclosing_y1 = torch.min(pred_y1, target_y1)
    enclosing_x2 = torch.max(pred_x2, target_x2)
    enclosing_y2 = torch.max(pred_y2, target_y2)
    enclosing_diagonal = (enclosing_x2 - enclosing_x1) ** 2 + (enclosing_y2 - enclosing_y1) ** 2

    # IOU Loss
    IoU_Loss = (1 - IoU)

    # CIoU
    distance_Loss = center_dist / enclosing_diagonal

    # Aspect ratio penalty
    v = (4 / (3.14159 ** 2)) * torch.pow(torch.atan(target_boxes[:, 2] / target_boxes[:, 3]) - torch.atan(pred_boxes[:, 2] / pred_boxes[:, 3]), 2)
    alpha = v / (1 - IoU + v) # Aspect ratio trade-off coeff
    AR_Loss = alpha * v

    ciou_loss = IoU_Loss +  AR_Loss + distance_Loss # CIoU loss
    return ciou_loss.mean()

