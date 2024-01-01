from ultralytics import YOLO
import torch
import cv2
import os


def save_image(image, path):
    """
    Save an image to a specified path.

    Parameters:
        image (ndarray): The image to be saved.
        path (str): The path where the image will be saved.
    """
    # Create the directory if it doesn't exist
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Save the image
    cv2.imwrite(path, image)

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = YOLO("/home/kasra/PycharmProjects/YOLOv8_customize/runs/detect/train/weights/best.pt")  # load a pretrained model (recommended for training)
img = '/home/kasra/kasra_files/data-shenasname/ai_files_20230726_1/2590454252_0.jpg'  # or file, Path, PIL, OpenCV, numpy, list
image = cv2.imread(img)
# Inference
img = '/home/kasra/PycharmProjects/YOLOv8_customize/extra_files/image2.jpg'

save_image(image, img)
results = model.predict(img, save=True, imgsz=640, conf=0.3, save_txt=True)
print(results[0].boxes.data)
# returns xyxy of bounding box + confidence and class number
# Results
# results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
