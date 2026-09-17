import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="Fine-tune a YOLO detector for CharmLens.")
parser.add_argument("--data", required=True, help="Path to YOLO data.yaml")
parser.add_argument("--model", default="yolo11s.pt", help="Base model or checkpoint")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--imgsz", type=int, default=960)
args = parser.parse_args()

model = YOLO(args.model)
model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, patience=25, cache=True, project="runs/charmlens", name="detector")
