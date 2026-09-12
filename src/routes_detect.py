"""
src/routes_detect.py -- Part A: POST /detect with optional device (cuda / cpu) selection
"""
import io
import time
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image

from .detector import get_detector

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post("/detect")
async def detect(
    image: UploadFile = File(...), 
    score_threshold: float = 0.5,
    device: Optional[str] = None
):
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {image.content_type}")

    raw = await image.read()
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image too large (max 10MB)")

    try:
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image file")

    detector = get_detector(device=device)
    start = time.time()
    detections = detector.predict(pil_image, score_threshold=score_threshold)
    latency_ms = round((time.time() - start) * 1000, 1)

    return {
        "image_id": image.filename,
        "device_used": detector.device.upper(),
        "detections": detections,
        "count": len(detections),
        "latency_ms": latency_ms,
    }
