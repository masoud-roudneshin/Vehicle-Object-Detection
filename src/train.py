import torch
import torch.optim as optim
from data_loader import get_data_loader
from model import load_yolo_model
from utils import bbox_ciou
from loss import YOLOv7Loss

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model, dataloader, num_epochs, lr):
  criterion = YOLOv7Loss()
  optimizer = optim.Adam(model.parameters(), lr = lr)

  for epoch in range(num_epochs):
    for images, targets in dataloader:
      images = images.to(device)
      targets = targets.to(device)
      outputs = model(images)

      loss = criterion(outputs, targets)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item(): 0.4f}")
