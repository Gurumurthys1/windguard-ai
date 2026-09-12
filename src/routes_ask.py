"""
src/routes_ask.py -- Part B: POST /ask
"""
import io
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image

from .detector import get_detector
from .reasoning import answer_question

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024


@router.post("/ask")
async def ask(
    question: str = Form(...),
    image: Optional[UploadFile] = File(None),
    min_confidence: float = Form(0.5),
):
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="`question` is required")

    pil_image = None
    if image is not None:
        if image.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(status_code=415, detail=f"Unsupported content type: {image.content_type}")
        raw = await image.read()
        if len(raw) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Image too large (max 10MB)")
        try:
            pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Could not decode image file")

    detector = get_detector()
    result = answer_question(question, pil_image, detector, min_confidence=min_confidence)
    return result
