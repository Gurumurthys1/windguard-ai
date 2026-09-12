import os
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from src.detector import get_detector
from src.reasoning import answer_question, classify_intent

# Load fine-tuned RT-DETR model (CPU mode for Hugging Face Spaces)
detector = get_detector(device="cpu")

CLASS_COLORS = {
    "crack": "#EF4444",           # Red
    "craze": "#F59E0B",           # Amber
    "hide_craze": "#8B5CF6",      # Purple
    "corrosion": "#EC4899",       # Pink
    "surface_injure": "#10B981",  # Green
    "thunderstrike": "#3B82F6",   # Blue
}

def draw_boxes_on_image(image: Image.Image, detections: list) -> Image.Image:
    """Draw bounding boxes and class labels directly onto the PIL Image."""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    for det in detections:
        box = det.get("bbox", [])
        if len(box) != 4:
            continue
        xmin, ymin, xmax, ymax = box
        label = det.get("class", "defect")
        score = det.get("confidence", 0.0)
        color = CLASS_COLORS.get(label, "#06B6D4")
        
        # Draw rectangle
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
        
        # Draw label tag
        text = f"{label} {score:.0%}"
        draw.rectangle([xmin, ymin - 16, xmin + len(text) * 8, ymin], fill=color)
        draw.text((xmin + 2, ymin - 15), text, fill="white")
        
    return img_copy

def analyze_turbine(image, question, conf_threshold):
    if image is None:
        return None, "⚠️ Please upload an image of a wind turbine blade."
    
    # 1. Run RT-DETR Prediction
    detections = detector.predict(image, score_threshold=conf_threshold)
    
    # 2. Draw detections on image
    annotated_img = draw_boxes_on_image(image, detections)
    
    # 3. Reasoning Layer
    if question and question.strip():
        reason_res = answer_question(question, image, detector, min_confidence=conf_threshold)
        summary = (
            f"### 🤖 AI Reasoning Analysis\n\n"
            f"**Answer:**\n{reason_res['answer']}\n\n"
            f"---\n"
            f"- **Used Detector:** `{'Yes' if reason_res.get('used_detector') else 'No'}`\n"
            f"- **Reasoning Trace:** *{reason_res.get('reasoning_trace', 'N/A')}*\n"
            f"- **Defects Found:** `{len(detections)}`"
        )
    else:
        counts = {}
        for d in detections:
            c = d["class"]
            counts[c] = counts.get(c, 0) + 1
        counts_str = ", ".join([f"{k}: {v}" for k, v in counts.items()]) if counts else "No defects detected"
        summary = (
            f"### 🔍 Detection Summary\n\n"
            f"- **Total Defects Found:** `{len(detections)}`\n"
            f"- **Defect Breakdown:** {counts_str}\n\n"
            f"💡 *Ask a question above to test the structured reasoning engine!*"
        )
        
    return annotated_img, summary

# Create Gradio Web UI
demo = gr.Interface(
    fn=analyze_turbine,
    inputs=[
        gr.Image(type="pil", label="📸 Upload Wind Turbine Blade Image"),
        gr.Textbox(
            label="💬 Natural Language Question",
            placeholder="e.g., How many cracks are visible? What repair procedure is needed for this blade?"
        ),
        gr.Slider(minimum=0.1, maximum=0.9, value=0.35, step=0.05, label="Detection Confidence Threshold"),
    ],
    outputs=[
        gr.Image(type="pil", label="🎯 Defect Detections"),
        gr.Markdown(label="📋 Reasoning Output"),
    ],
    title="🌪️ WindGuard AI — RT-DETR Defect Detection & Reasoning API",
    description="Fine-tuned RT-DETR Object Detection and Hand-Written Reasoning for Industrial Wind Turbine Blade Inspection.",
)

if __name__ == "__main__":
    demo.launch()
