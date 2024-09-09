
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import cv2
import os

class CustomDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform = None):
      self.annotations = pd.read_scv(csv_file)
      self.image_dir = image_dir
      self.transform = transform
    def __len__(self):
      return len(self.annotations)
    def __getitem__(self, index):
      image_path = os.path.join(self.image_dir, self.annotations.iloc[index, 0]) #f"{self.image_dir}/{self.annotations.iloc[index,0]}"
      image = Image.open(image_path).convert("RGB") # cv2.imread(image_path)
      boxes = self.annotations.iloc[index, 1:].values
      boxes = torch.tensor([float(b) for b in boxes]).reshape(-1,5) #[class, x_center, y_center, w, h]

      if self.transform:
          image = self.transform(image)

    return image, boxes

def get_data_loader(csv_file, image_dir,batch_size, transform = None):
  dataset = CustomDataset(csv_file, image_dir, transform = None)
  return DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = 2)
