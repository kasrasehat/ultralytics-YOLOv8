from ultralytics import YOLO


model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)
model.train(data='/home/kasra/PycharmProjects/ultralytics/config.yaml', epochs=70, imgsz=640)
model.val()
