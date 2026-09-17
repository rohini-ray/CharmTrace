from __future__ import annotations

import base64
import os
from functools import lru_cache

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = FastAPI(title="CharmLens")


@lru_cache(maxsize=1)
def detector():
    """Load once so image requests do not repeatedly initialize the model."""
    try:
        from ultralytics import YOLO
        return YOLO(os.getenv("CHARMLENS_MODEL", os.getenv("VISIONTRACE_MODEL", "yolo11n.pt")))
    except Exception:
        return None


def image_data_url(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not encode result image")
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


def find_blobs(bgr: np.ndarray, min_area: int):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 55, 160)
    threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 31, 3)
    mask = cv2.bitwise_or(threshold, cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        moments = cv2.moments(contour)
        if not moments["m00"]:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        blobs.append({
            "bbox": [x, y, w, h], "area": round(float(area), 1),
            "perimeter": round(float(perimeter), 1),
            "centroid": [round(moments["m10"] / moments["m00"], 1), round(moments["m01"] / moments["m00"], 1)],
            "circularity": round(float(4 * np.pi * area / (perimeter * perimeter)) if perimeter else 0, 3),
            "contour": contour,
        })
    return edges, mask, sorted(blobs, key=lambda blob: blob["area"], reverse=True)


@app.post("/api/analyze")
async def analyze(image: UploadFile = File(...), confidence: float = Form(0.35), min_area: int = Form(250)):
    raw = await image.read()
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "Please upload a valid image file.")
    if not 0.05 <= confidence <= 0.95 or not 20 <= min_area <= 100000:
        raise HTTPException(400, "Analysis controls are outside supported bounds.")

    edges, mask, blobs = find_blobs(bgr, min_area)
    output = bgr.copy()
    for index, blob in enumerate(blobs, 1):
        cv2.drawContours(output, [blob["contour"]], -1, (34, 211, 238), 2)
        x, y, w, h = blob["bbox"]
        cv2.putText(output, f"B{index} {blob['area']:.0f}px", (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, .48, (34, 211, 238), 2)
        del blob["contour"]

    detections = []
    model = detector()
    if model is not None:
        try:
            result = model.predict(bgr, conf=confidence, verbose=False)[0]
            names = result.names
            for box in result.boxes:
                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                score = float(box.conf[0])
                label = names[int(box.cls[0])]
                detections.append({"label": label, "confidence": round(score, 3), "bbox": [x1, y1, x2 - x1, y2 - y1]})
                cv2.rectangle(output, (x1, y1), (x2, y2), (16, 185, 129), 2)
                cv2.putText(output, f"{label} {score:.0%}", (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, .55, (16, 185, 129), 2)
        except Exception:
            pass

    return {"annotated": image_data_url(output), "edges": image_data_url(edges), "mask": image_data_url(mask),
            "blobs": blobs, "detections": detections, "model_ready": model is not None}


app.mount("/", StaticFiles(directory=os.path.join(ROOT, "frontend"), html=True), name="frontend")
