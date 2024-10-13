from ultralytics import YOLO
import torch

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

if __name__ == "__main__":
    # Load a model
    model = YOLO("yolov8s.pt")  
    # load a pretrained YOLOv8n model

    # Train the model with  GPU
    results = model.train(
        data="data.yaml", 
        epochs=200, 
        imgsz=640, 
        device='0',  # use  GPU
        project='runs', 
        name='trainC'
    )

