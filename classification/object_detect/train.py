import warnings
import os

warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("ultralytics/cfg/models/11/yolo11m-obb.yaml")
    # model.load('yolov11.pt') # loading pretrain weights
    model.train(
        data=os.path.join(os.path.dirname(__file__), "data.yaml"),
        cache=False,
        imgsz=640,
        epochs=1000,
        batch=64,
        close_mosaic=10,
        device="0",
        optimizer="SGD",  # using SGD
        project=os.path.join(os.path.dirname(__file__), "runs/train"),
        name="exp",
    )
