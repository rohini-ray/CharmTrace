# CharmLens

CharmLens is a local, explainable image-analysis tool for finding keychains and tiny charms in busy photos. Upload a picture, tune the scan, and get an annotated trace alongside the edge map, candidate mask, and measurable region details.

## Highlights

- **Charm-first workflow:** the interface is designed around finding keychains, not generic computer-vision output.
- **Explainable results:** inspect candidate blobs with area, centroid, perimeter, circularity, and a visual mask.
- **Optional trained detector:** when YOLO weights are available, confirmed detections are drawn in mint green beside the coral candidate regions.
- **Private by default:** the FastAPI server runs locally and uploaded images are processed in memory.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.app:app --reload
```

Open <http://127.0.0.1:8000>, upload a PNG, JPG, or WEBP image, then adjust detection confidence and the smallest charm area before selecting **Find keychains**.

## Optional custom model

CharmLens will use `CHARMLENS_MODEL` when it points to a trained Ultralytics checkpoint. If no usable weights are available, the candidate-region analysis still works and the model status is shown as **OFF**.

```powershell
$env:CHARMLENS_MODEL = 'C:\path\to\best.pt'
uvicorn backend.app:app --reload
```

To train a detector, prepare a YOLO dataset:

```text
dataset/
  images/train, images/val
  labels/train, labels/val
  data.yaml
```

Then run:

```powershell
python backend/train.py --data C:\path\to\data.yaml --epochs 100 --model yolo11s.pt
```

The resulting checkpoint is written under `runs/charmlens/`.

## API

`POST /api/analyze` accepts multipart fields `image`, `confidence`, and `min_area`. It returns data URLs for `annotated`, `edges`, and `mask`, plus `blobs`, `detections`, and `model_ready` metadata.

## Project layout

- `frontend/` — CharmLens single-page interface.
- `backend/app.py` — FastAPI endpoint and OpenCV analysis pipeline.
- `backend/train.py` — optional YOLO training entry point.