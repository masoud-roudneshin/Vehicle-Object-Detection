
if __name__ == "__main__":
  # Give csv_file
  # Give image_dir
  batch_size = 16
  num_epochs = 20
  lr = 0.001
  num_classes = 20

  dataloader = get_data_loader(csv_file, image_dir, batch_size)
  model = load_yolo_model(num_classes).cuda()

  train(model, dataloader, num_epochs, lr)
