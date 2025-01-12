from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO(r"C:\Users\ABHIRAM\Downloads\best.pt")

# Use the model to predict from webcam (source=0)
model.predict(source=0, show=True, conf=0.5)

