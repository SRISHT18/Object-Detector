"""
AI Object Detector — FastAPI Backend
Uses YOLOv8 (ultralytics) to detect objects in uploaded images.

Run with:
    pip install -r requirements.txt
    python main.py
    
Then open:  http://localhost:8000
"""

import io
import base64
import os
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import uvicorn

app = FastAPI(title="AI Object Detector", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── serve index.html at root ─────────────────────────────────────────────────
INDEX = Path(__file__).parent / "index.html"

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    if INDEX.exists():
        return HTMLResponse(INDEX.read_text())
    return HTMLResponse("<h2>Put index.html next to main.py and restart.</h2>")

# ── load model ───────────────────────────────────────────────────────────────
model: YOLO | None = None

@app.on_event("startup")
async def load_model():
    global model
    print("Loading YOLOv8n model…")
    model = YOLO("yolov8n.pt")
    print("Model ready ✓  →  open http://localhost:8000")

# ── colour palette ───────────────────────────────────────────────────────────
PALETTE = [
    (220, 80,  60),  (60,  160, 220), (60,  200, 120), (220, 170, 40),
    (180, 60,  220), (60,  210, 210), (220, 110, 180), (140, 200, 60),
    (220, 140, 60),  (80,  120, 220),
]

def colour_for(class_id: int) -> tuple:
    return PALETTE[class_id % len(PALETTE)]

# ── /detect ──────────────────────────────────────────────────────────────────
@app.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    confidence: float = 0.35,
) -> JSONResponse:
    if model is None:
        raise HTTPException(503, "Model not loaded yet — retry in a moment.")

    raw = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not open image. Upload a valid JPEG/PNG.")

    img_np  = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    results = model(img_np, conf=confidence, verbose=False)[0]
    detections: List[Dict[str, Any]] = []

    for box in results.boxes:
        cls_id     = int(box.cls[0])
        conf_score = float(box.conf[0])
        label      = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        colour     = colour_for(cls_id)
        cbgr       = (colour[2], colour[1], colour[0])

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), cbgr, 2)
        tag = f"{label}  {conf_score:.0%}"
        (tw, th), bl = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img_bgr, (x1, y1 - th - bl - 6), (x1 + tw + 8, y1), cbgr, -1)
        cv2.putText(img_bgr, tag, (x1 + 4, y1 - bl - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        detections.append({
            "label":      label,
            "confidence": round(conf_score, 4),
            "box":        {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "color":      f"rgb({colour[0]},{colour[1]},{colour[2]})",
        })

    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(buf).decode()

    label_counts: Dict[str, int] = {}
    for d in detections:
        label_counts[d["label"]] = label_counts.get(d["label"], 0) + 1

    return JSONResponse({
        "detections":   detections,
        "label_counts": label_counts,
        "total":        len(detections),
        "image_b64":    b64,
        "image_size":   {"width": pil_img.width, "height": pil_img.height},
    })

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

# ── entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import webbrowser, threading, time
    def _open():
        time.sleep(2)
        webbrowser.open("http://localhost:8000")
    threading.Thread(target=_open, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
