import torch
import cv2
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def detect_objects(image_path):
    """Detects objects in an image using YOLOv5."""
    img = cv2.imread(image_path)
    results = model(img)
    results.show()  # Display detected objects

if __name__ == "__main__":
    detect_objects("objects.jpg")
