"""
src/detector.py -- thin wrapper around the fine-tuned RT-DETR model.
Configured default runtime device: CPU engine.
"""
import os
import torch
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

MODEL_DIR = os.environ.get("WINDGUARD_MODEL_DIR", "models/rtdetr-wtbd-final")
DEFAULT_SCORE_THRESHOLD = float(os.environ.get("WINDGUARD_SCORE_THRESHOLD", "0.5"))
DEVICE_ENV = os.environ.get("WINDGUARD_DEVICE", "cpu")


class Detector:
    def __init__(self, model_dir: str = MODEL_DIR, device: str = None):
        self.processor = RTDetrImageProcessor.from_pretrained(model_dir)
        self.model = RTDetrForObjectDetection.from_pretrained(model_dir)
        self.model.eval()
        
        # Configure runtime device (CPU mode default)
        self.device = device or DEVICE_ENV
        self.model.to(self.device)
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def predict(self, image: Image.Image, score_threshold: float = DEFAULT_SCORE_THRESHOLD):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=score_threshold
        )[0]

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            xmin, ymin, xmax, ymax = [round(v, 1) for v in box.tolist()]
            detections.append({
                "class": self.id2label[int(label.item())],
                "confidence": round(float(score.item()), 4),
                "bbox": [xmin, ymin, xmax, ymax],
            })
        return detections


_detector_singleton = None


def get_detector(device: str = None) -> Detector:
    global _detector_singleton
    if _detector_singleton is None or (device and _detector_singleton.device != device):
        _detector_singleton = Detector(device=device or "cpu")
    return _detector_singleton
