
import torch
import torch.nn as nn
from models.experimental import attempt_load

class YOLOV7(nn.Module):
    def __init__(self, num_classes, weights_path = "yolov7.pt"):
        super(YOLOV7, self).__init__()
        self.model = attempt_load(weights_path, map = "cuda")

        for params in self.model.parameters():
          params.requires_grad = False

        self.model.model[-1].nc = num_classes
        self.model.model[-1].no = num_classes + 5

        self.model.model[-1].reset_parameters()

    def forward(self, x):
        return self.model(x)

def load_yolo_model(num_classes, weights_path = "yolov7.pt"):
    return YOLOV7(num_classes = num_classes, weights_path = weights_path)
