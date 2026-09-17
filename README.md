# VisionTrace

VisionTrace is a local web application for accurate **edge-aware blob analysis** and **object detection** in images. It combines Canny edges, adaptive thresholding, morphology, contour filtering, and non-maximum-suppressed YOLO detections in one reviewable result.

## What it does

- Finds blob regions with area, centroid, perimeter, and circularity metrics.
- Draws edge maps and blob masks so results are explainable, not a black box.
- Runs YOLO object detection when `ultralytics` weights are available (the first use downloads the default model).
- Provides a training entry point for a labelled YOLO-format dataset.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.app:app --reload
```

Open `http://127.0.0.1:8000`, upload an image, then tune the confidence and minimum-area controls.

## Train a detector

Prepare a YOLO dataset and a `data.yaml`, for example:

```text
dataset/
  images/train, images/val
  labels/train, labels/val
  data.yaml
```

Run:

```powershell
python backend/train.py --data C:\path\to\data.yaml --epochs 100 --model yolo11s.pt
```

Use the resulting `best.pt` path in the `VISIONTRACE_MODEL` environment variable before starting the server. For real accuracy, label representative images, reserve a validation set, and use the model size and epoch count appropriate for the available GPU. Do not evaluate on training images alone.

## API

`POST /api/analyze` accepts `image`, `confidence`, and `min_area`, returning an annotated PNG plus detection and blob metadata.
