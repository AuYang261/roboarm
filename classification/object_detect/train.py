import warnings
import os

warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11x-obb.pt")
    # model.load(
    #     os.path.join(
    #         os.path.dirname(__file__), "runs", "train", "exp2", "weights", "best.pt"
    #     )
    # )
    model.train(
        data=os.path.join(os.path.dirname(__file__), "data.yaml"),
        cache=False,
        imgsz=640,
        epochs=1000,
        batch=32,
        close_mosaic=10,
        device="0",
        optimizer="SGD",  # using SGD
        project=os.path.join(os.path.dirname(__file__), "runs/train"),
        name="exp",
    )
