import os
import gradio as gr
from PIL import Image
import numpy as np

from src.detector import get_detector
from src.reasoning import ask_reasoning_engine

# Initialize RT-DETR model detector
detector = get_detector()

def analyze_blade(image, question, conf_threshold):
    if image is None:
        return None, "⚠️ Please upload an image of a wind turbine blade."
    
    # 1. Run RT-DETR Detection
    boxes, labels, scores, latency = detector.predict(image, conf_threshold=conf_threshold)
    
    # 2. Draw Bounding Boxes with distinct class colors
    annotated = detector.draw_detections(image, boxes, labels, scores)
    
    # 3. Format detections for reasoning
    detections = [
        {"class_name": l, "confidence": float(s), "box_2d": [int(x) for x in b]}
        for b, l, s in zip(boxes, labels, scores)
    ]
    
    if question and question.strip():
        res = ask_reasoning_engine(question, detections)
        summary = (
            f"### 🤖 AI Reasoning Analysis\n\n"
            f"**Answer:** {res['answer']}\n\n"
            f"---  \n"
            f"- **Confidence Score:** `{res['confidence']:.2f}`\n"
            f"- **Intent Category:** `{res['category']}`\n"
            f"- **Inference Latency:** `{latency:.1f} ms`\n"
            f"- **Total Defects Detected:** `{len(boxes)}`"
        )
    else:
        counts = {}
        for l in labels:
            counts[l] = counts.get(l, 0) + 1
        counts_str = ", ".join([f"{k}: {v}" for k, v in counts.items()]) if counts else "None"
        summary = (
            f"### 🔍 Detection Summary\n\n"
            f"- **Defects Detected:** `{len(boxes)}` ({counts_str})\n"
            f"- **Inference Latency:** `{latency:.1f} ms`\n\n"
            f"💡 *Tip: Type a natural-language question in the prompt box to trigger the reasoning engine!*"
        )
        
    return annotated, summary

# Build Gradio Demo Interface
demo = gr.Interface(
    fn=analyze_blade,
    inputs=[
        gr.Image(type="pil", label="📸 Upload Wind Turbine Blade Image"),
        gr.Textbox(
            label="💬 Ask a Natural Language Question",
            placeholder="e.g., How many defects are there? What is the most severe damage? Is there lightning puncture?"
        ),
        gr.Slider(minimum=0.1, maximum=0.9, value=0.25, step=0.05, label="Detection Confidence Threshold"),
    ],
    outputs=[
        gr.Image(type="pil", label="🎯 RT-DETR Detection Output"),
        gr.Markdown(label="📋 Structured Reasoning Output"),
    ],
    title="🌪️ WindGuard AI — RT-DETR Defect Detection & Reasoning API",
    description=(
        "### Industrial Wind Turbine Blade Defect Inspection & Reasoning\n"
        "Powered by fine-tuned **RT-DETR (ResNet-50vd)** on domain-specific wind turbine blade defects "
        "and a framework-free structured reasoning engine (No LangChain/CrewAI)."
    ),
    examples=[
        ["docs/sample_annotations.png", "What defects are present on this turbine blade?", 0.25]
    ] if os.path.exists("docs/sample_annotations.png") else None,
)

if __name__ == "__main__":
    demo.launch()
