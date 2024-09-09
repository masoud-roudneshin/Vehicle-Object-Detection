
import torch
import torch.nn as nn

class YOLOv7Loss(nn.Module):
    def __init__(self, num_classes, lambda_coord=5.0, lambda_noobj=0.5):
        super(YOLOv7Loss, self).__init__()
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord  # Weight for the bounding box regression
        self.lambda_noobj = lambda_noobj  # Weight for no-object confidence

        self.mse_loss = nn.MSELoss()  # Mean Squared Error for regression
        self.bce_loss = nn.BCEWithLogitsLoss()  # Binary Cross Entropy for objectness and classification

    def forward(self, pred, target):
        """
        Compute the YOLOv7 loss.

        :param pred: Predicted output from the model, [B, C, H, W]
                     Contains bounding boxes, objectness, and class predictions.
                     Format: [B, num_anchors * (5 + num_classes), H, W]
        :param target: Ground truth labels [B, num_anchors, 5 + num_classes] per bounding box
                       Format: [B, num_anchors, 5 + num_classes]
                       where 5 represents [x_center, y_center, width, height, objectness]
        :return: Total loss (bounding box + objectness + classification loss)
        """
        # Split the predictions into their respective parts
        pred_boxes = pred[..., :4]  # [x_center, y_center, w, h]
        pred_objectness = pred[..., 4]  # object confidence score
        pred_class_probs = pred[..., 5:]  # class probabilities

        # Split the target into respective parts
        target_boxes = target[..., :4]  # Ground truth [x_center, y_center, w, h]
        target_objectness = target[..., 4]  # Ground truth object confidence
        target_class_probs = target[..., 5:]  # Ground truth class probabilities

        # Compute the bounding box loss (regression)
        box_loss = self.lambda_coord * self.mse_loss(pred_boxes, target_boxes)

        # Compute objectness loss
        objectness_loss = self.bce_loss(pred_objectness, target_objectness)

        # Compute classification loss
        class_loss = self.bce_loss(pred_class_probs, target_class_probs)

        # Loss for no-object prediction (higher penalty when there's no object in the cell)
        no_object_loss = self.lambda_noobj * self.bce_loss(pred_objectness, torch.zeros_like(pred_objectness))

        # Total loss is the sum of all losses
        total_loss = box_loss + objectness_loss + class_loss + no_object_loss

        return total_loss

