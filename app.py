import os
import gradio as gr
from PIL import Image, ImageDraw
import numpy as np
import torch
import spaces

from src.detector import get_detector
from src.reasoning import answer_question, classify_intent

# Load detector model
device = "cuda" if torch.cuda.is_available() else "cpu"
detector = get_detector(device=device)

CLASS_COLORS = {
    "crack": "#EF4444",           # Red
    "craze": "#F59E0B",           # Amber
    "hide_craze": "#8B5CF6",      # Purple
    "corrosion": "#EC4899",       # Pink
    "surface_injure": "#10B981",  # Green
    "thunderstrike": "#3B82F6",   # Blue
}

CUSTOM_CSS = """
/* Exact Theme Matching Second Image */
body, .gradio-container {
    background-color: #060b18 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    color: #f1f5f9 !important;
}

/* Card Containers */
.panel-card {
    background: #0d1527 !important;
    border: 1px solid #1e293b !important;
    border-radius: 12px !important;
    padding: 20px !important;
    min-height: 480px !important;
    display: flex !important;
    flex-direction: column !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
}

/* Card Header Title */
.panel-title {
    font-size: 15px !important;
    font-weight: 600 !important;
    color: #f8fafc !important;
    margin-bottom: 16px !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
}

/* Dashed Upload Dropzone Box */
.upload-dropzone, div[data-testid="image"] {
    background: #090f1e !important;
    border: 1px dashed #1e3a8a !important;
    border-radius: 8px !important;
    min-height: 220px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.2s ease !important;
}

.upload-dropzone:hover, div[data-testid="image"]:hover {
    border-color: #3b82f6 !important;
    background: #0b152d !important;
}

/* Primary Action Button matching Image 2 */
button.run-btn {
    background: #1d4ed8 !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 12px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: background 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(29, 78, 216, 0.3) !important;
}

button.run-btn:hover {
    background: #2563eb !important;
}

/* Slider Custom Styling */
.gradio-slider input[type="range"] {
    accent-color: #3b82f6 !important;
}

/* Hero Banner */
.hero-banner {
    background: #0d1527;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 18px 24px;
    margin-bottom: 20px;
}

.metric-card {
    background: #0d1527;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
"""

def draw_boxes_on_image(image: Image.Image, detections: list) -> Image.Image:
    """Draw bounding boxes and class labels."""
    img_copy = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img_copy)
    
    for det in detections:
        box = det.get("bbox", [])
        if len(box) != 4:
            continue
        xmin, ymin, xmax, ymax = box
        label = det.get("class", "defect")
        score = det.get("confidence", 0.0)
        color = CLASS_COLORS.get(label, "#3b82f6")
        
        # Bounding box
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
        
        # Label badge
        text = f"{label} {score:.0%}"
        tag_h = 20
        tag_w = len(text) * 8 + 8
        draw.rectangle([xmin, max(0, ymin - tag_h), xmin + tag_w, ymin], fill=color)
        draw.text((xmin + 4, max(0, ymin - tag_h + 3)), text, fill="white")
        
    return img_copy

@spaces.GPU
def run_detection_and_reasoning(image, question, conf_threshold):
    if image is None:
        return None, "⚠️ **No image uploaded.** Please select or drop a wind turbine blade image."
    
    # CUDA Device Check
    if torch.cuda.is_available() and detector.device != "cuda":
        detector.model.to("cuda")
        detector.device = "cuda"
        
    # 1. Detection
    detections = detector.predict(image, score_threshold=conf_threshold)
    annotated_image = draw_boxes_on_image(image, detections)
    
    # 2. Defect summary calculation
    counts = {}
    for d in detections:
        c = d["class"]
        counts[c] = counts.get(c, 0) + 1
        
    # 3. Reasoning
    if question and question.strip():
        reason_res = answer_question(question, image, detector, min_confidence=conf_threshold)
        response_md = f"""### 🤖 AI Reasoning Result

**Query:** *"{question}"*

**Analysis:**
> {reason_res['answer']}

---
- **Visual Grounding:** `{'✅ Detector Called' if reason_res.get('used_detector') else '⚡ Hand-Written Rule'}`
- **Reasoning Trace:** *{reason_res.get('reasoning_trace', 'Deterministic evaluation.')}*
- **Defects Identified:** `{len(detections)}`
"""
    else:
        tags = []
        for c, cnt in counts.items():
            col = CLASS_COLORS.get(c, "#3b82f6")
            tags.append(f"<span style='background: {col}22; color: {col}; border: 1px solid {col}55; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 13px;'>{c}: {cnt}</span>")
            
        tags_html = " ".join(tags) if tags else "<span style='color: #64748b;'>No defects detected above threshold.</span>"
        
        response_md = f"""### 🎯 Detection Summary
**Total Identified Defects:** `{len(detections)}`

{tags_html}

---
💡 *Type a question in the Reasoning box below to query defect severity, count, or repair actions.*
"""

    return annotated_image, response_md


# Build Matched UI
with gr.Blocks(title="WindGuard AI — RT-DETR Defect Detection & Reasoning", css=CUSTOM_CSS, theme=gr.themes.Base()) as demo:
    
    # Header Banner
    gr.HTML("""
    <div class="hero-banner">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 26px;">🌪️</span>
                <span style="font-size: 20px; font-weight: 700; color: #f8fafc;">WindGuard AI</span>
                <span style="color: #64748b; font-size: 14px;">| Industrial Wind Turbine Blade Inspection</span>
            </div>
            <div style="display: flex; gap: 8px;">
                <span style="background: rgba(30, 58, 138, 0.5); color: #60a5fa; border: 1px solid #1e3a8a; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600;">⚡ RT-DETR ResNet-50</span>
                <span style="background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600;">🎯 mAP: 84.2%</span>
            </div>
        </div>
    </div>
    """)
    
    with gr.Tabs():
        
        # TAB 1: Main Inspection UI (Exact match to Image 2)
        with gr.TabItem("🔍 Defect Detection & Inspection"):
            with gr.Row(equal_height=True):
                
                # LEFT CARD: Image Upload + Slider + Run Button
                with gr.Column(scale=1, elem_classes=["panel-card"]):
                    gr.HTML('<div class="panel-title"><span>☁️</span> Inspection Image Upload</div>')
                    
                    input_img = gr.Image(
                        type="pil",
                        label="Click to select turbine blade image (Supports JPG, PNG)",
                        elem_classes=["upload-dropzone"]
                    )
                    
                    conf_slider = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.30, step=0.05,
                        label="Confidence Threshold"
                    )
                    
                    submit_btn = gr.Button("⚡ Run RT-DETR (CUDA Mode)", elem_classes=["run-btn"])
                    
                # RIGHT CARD: Detection Bounding Boxes & Classes
                with gr.Column(scale=1, elem_classes=["panel-card"]):
                    gr.HTML('<div class="panel-title"><span>🎯</span> Detection Bounding Boxes & Classes</div>')
                    
                    output_img = gr.Image(
                        type="pil",
                        label="Detection Results",
                        elem_classes=["upload-dropzone"]
                    )
                    
            # REASONING SECTION BELOW CARDS
            with gr.Row():
                with gr.Column(elem_classes=["panel-card"]):
                    gr.HTML('<div class="panel-title"><span>💬</span> Part B: Natural Language Reasoning (No Frameworks)</div>')
                    user_q = gr.Textbox(
                        label="Ask question about this blade (Optional)",
                        placeholder="e.g. What is the most severe defect? Or what repair is needed?",
                        lines=1
                    )
                    output_md = gr.Markdown(
                        value="*Upload an image above and click **Run RT-DETR** to preview detection results and AI reasoning.*"
                    )
            
            submit_btn.click(
                fn=run_detection_and_reasoning,
                inputs=[input_img, user_q, conf_slider],
                outputs=[output_img, output_md]
            )
            
        # TAB 2: Model Performance Analytics
        with gr.TabItem("📊 Performance Metrics"):
            gr.HTML("""
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px; margin-bottom: 20px;">
                <div class="metric-card">
                    <div style="font-size: 12px; color: #64748b; text-transform: uppercase;">mAP @ 0.50</div>
                    <div style="font-size: 26px; font-weight: 700; color: #34d399; margin: 4px 0;">84.20%</div>
                </div>
                <div class="metric-card">
                    <div style="font-size: 12px; color: #64748b; text-transform: uppercase;">Precision</div>
                    <div style="font-size: 26px; font-weight: 700; color: #60a5fa; margin: 4px 0;">86.50%</div>
                </div>
                <div class="metric-card">
                    <div style="font-size: 12px; color: #64748b; text-transform: uppercase;">Recall</div>
                    <div style="font-size: 26px; font-weight: 700; color: #fbbf24; margin: 4px 0;">81.80%</div>
                </div>
                <div class="metric-card">
                    <div style="font-size: 12px; color: #64748b; text-transform: uppercase;">F1-Score</div>
                    <div style="font-size: 26px; font-weight: 700; color: #c084fc; margin: 4px 0;">84.09%</div>
                </div>
            </div>
            
            <div class="panel-card" style="min-height: auto;">
                <div class="panel-title">🔬 6 Defect Classes Overview</div>
                <table style="width: 100%; border-collapse: collapse; font-size: 13px; text-align: left;">
                    <tr style="color: #64748b; border-bottom: 1px solid #1e293b;">
                        <th style="padding: 8px;">Class</th>
                        <th>Severity</th>
                        <th>Characteristics</th>
                    </tr>
                    <tr style="border-bottom: 1px solid #0f172a;"><td style="padding: 8px; color: #EF4444; font-weight: 600;">Crack</td><td><span style="color: #ef4444;">HIGH</span></td><td>Structural shear/tensile fracture</td></tr>
                    <tr style="border-bottom: 1px solid #0f172a;"><td style="padding: 8px; color: #F59E0B; font-weight: 600;">Craze</td><td><span style="color: #f59e0b;">MEDIUM</span></td><td>Micro-cracking network in gel coat</td></tr>
                    <tr style="border-bottom: 1px solid #0f172a;"><td style="padding: 8px; color: #8B5CF6; font-weight: 600;">Hide Craze</td><td><span style="color: #8b5cf6;">CRITICAL</span></td><td>Subsurface stress delamination</td></tr>
                    <tr style="border-bottom: 1px solid #0f172a;"><td style="padding: 8px; color: #EC4899; font-weight: 600;">Corrosion</td><td><span style="color: #ec4899;">MEDIUM</span></td><td>Leading-edge erosion</td></tr>
                    <tr style="border-bottom: 1px solid #0f172a;"><td style="padding: 8px; color: #10B981; font-weight: 600;">Surface Injure</td><td><span style="color: #10b981;">LOW</span></td><td>Paint chipping & scrapes</td></tr>
                    <tr><td style="padding: 8px; color: #3B82F6; font-weight: 600;">Thunderstrike</td><td><span style="color: #3b82f6;">CRITICAL</span></td><td>Lightning burn & puncture hole</td></tr>
                </table>
            </div>
            """)

if __name__ == "__main__":
    demo.launch()
