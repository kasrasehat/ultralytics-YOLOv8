from ultralytics import YOLO
import torch


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
results = model.train(data="/home/kasra/PycharmProjects/ultralytics/config.yaml", epochs=60)
model.val()
