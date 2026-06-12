import numpy as np
import pandas as pd
!pip install -q ultralytics
import torch
print("GPU Available:", torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using:", device)

from ultralytics import YOLO

model = YOLO("yolo11s.pt")

results = model.train(
    data="/kaggle/input/datasets/rahilvats/football-dataset/data.yaml",
    epochs=50,
    imgsz=1280,
    batch=8,
    device=[0,1],
    workers=2
)


model.val()


preds = model.predict(
    source="/kaggle/input/datasets/rahilvats/football-dataset/valid/images",
    save=True
)

print("Training complete. Check /kaggle/working/runs/detect/train")
