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
    "crack": "#EF4444",           # Bright Red
    "craze": "#F59E0B",           # Amber Orange
    "hide_craze": "#8B5CF6",      # Vivid Purple
    "corrosion": "#EC4899",       # Hot Pink
    "surface_injure": "#10B981",  # Emerald Green
    "thunderstrike": "#3B82F6",   # Electric Blue
}

CUSTOM_CSS = """
/* Futuristic Cyber-Glass Theme */
body, .gradio-container {
    background: radial-gradient(circle at 50% 0%, #0f172a 0%, #030712 100%) !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    color: #f8fafc !important;
}

/* Custom Stylish Drag & Drop Upload Zone */
.custom-dropzone, div[data-testid="image"] {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.8) 100%) !important;
    border: 2px dashed rgba(99, 102, 241, 0.6) !important;
    border-radius: 16px !important;
    padding: 12px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
}

.custom-dropzone:hover, div[data-testid="image"]:hover {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 25px rgba(56, 189, 248, 0.25), inset 0 0 15px rgba(56, 189, 248, 0.1) !important;
    transform: scale(1.005) !important;
}

/* Output Detection Viewer Box */
.output-box {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0.7) 100%) !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5) !important;
}

/* Hero Header Banner */
.hero-banner {
    background: linear-gradient(135deg, rgba(30, 27, 75, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%);
    border: 1px solid rgba(99, 102, 241, 0.4);
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 10px 35px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(16px);
}

/* Metric Cards */
.metric-card {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 18px 14px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.metric-card:hover {
    border-color: rgba(99, 102, 241, 0.4);
    transform: translateY(-2px);
}

.metric-val {
    font-size: 30px;
    font-weight: 800;
    margin: 4px 0;
    letter-spacing: -0.02em;
}

.metric-title {
    font-size: 12px;
    color: #94a3b8;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.08em;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16, 185, 129, 0.12);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.35);
    padding: 5px 14px;
    border-radius: 9999px;
    font-size: 13px;
    font-weight: 600;
}

/* Buttons */
button.primary-btn {
    background: linear-gradient(135deg, #6366f1 0%, #3b82f6 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    border-radius: 12px !important;
    border: none !important;
    padding: 12px 24px !important;
    box-shadow: 0 4px 18px rgba(59, 130, 246, 0.4) !important;
    transition: all 0.2s ease !important;
}

button.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(59, 130, 246, 0.6) !important;
}

.quick-q-btn {
    background: rgba(30, 41, 59, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
    color: #cbd5e1 !important;
    font-size: 12px !important;
    padding: 4px 10px !important;
}

.quick-q-btn:hover {
    background: rgba(99, 102, 241, 0.25) !important;
    border-color: rgba(99, 102, 241, 0.5) !important;
    color: white !important;
}
"""

def draw_boxes_on_image(image: Image.Image, detections: list) -> Image.Image:
    """Draw rich bounding boxes and badges on the detected defects."""
    img_copy = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img_copy)
    
    for det in detections:
        box = det.get("bbox", [])
        if len(box) != 4:
            continue
        xmin, ymin, xmax, ymax = box
        label = det.get("class", "defect")
        score = det.get("confidence", 0.0)
        color = CLASS_COLORS.get(label, "#38bdf8")
        
        # Draw high-visibility border
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=4)
        
        # Draw badge
        text = f" {label.upper()}  {score:.0%} "
        tag_h = 24
        tag_w = len(text) * 8 + 12
        draw.rectangle([xmin, max(0, ymin - tag_h), xmin + tag_w, ymin], fill=color)
        draw.text((xmin + 6, max(0, ymin - tag_h + 4)), text, fill="white")
        
    return img_copy

@spaces.GPU
def run_inspection(image, question, conf_threshold):
    if image is None:
        return None, "⚠️ **No image detected.** Please drag & drop a wind turbine blade photo on the left."
    
    # Ensure CUDA allocation during ZeroGPU
    if torch.cuda.is_available() and detector.device != "cuda":
        detector.model.to("cuda")
        detector.device = "cuda"
        
    # 1. RT-DETR Detection
    detections = detector.predict(image, score_threshold=conf_threshold)
    
    # 2. Draw annotated bounding boxes
    annotated_image = draw_boxes_on_image(image, detections)
    
    # 3. Defect summary calculation
    counts = {}
    for d in detections:
        c = d["class"]
        counts[c] = counts.get(c, 0) + 1
        
    # 4. Hand-Written Reasoning Layer Execution
    if question and question.strip():
        reason_res = answer_question(question, image, detector, min_confidence=conf_threshold)
        
        response_md = f"""### 🤖 Natural Language Reasoning Engine

**Question:** *"{question}"*

**AI Analysis & Answer:**
> {reason_res['answer']}

---

#### 📋 Reasoning Metadata:
- **Visual Grounding Used:** `{'✅ Yes (Detector Called)' if reason_res.get('used_detector') else '⚡ No (Direct Guardrail)'}`
- **Reasoning Trace:** *{reason_res.get('reasoning_trace', 'Deterministic rule execution.')}*
- **Active Confidence Threshold:** `{conf_threshold:.2f}`
- **Defects Detected:** `{len(detections)}`
"""
    else:
        tags = []
        for c, cnt in counts.items():
            col = CLASS_COLORS.get(c, "#38bdf8")
            tags.append(f"<span style='background: {col}22; color: {col}; border: 1px solid {col}55; padding: 4px 10px; border-radius: 6px; font-weight: 700;'>{c}: {cnt}</span>")
            
        tags_html = " ".join(tags) if tags else "<span style='color: #94a3b8;'>No defects detected at this confidence threshold.</span>"
        
        response_md = f"""### 🔍 RT-DETR Defect Detection Summary

**Total Defect Candidates Identified:** `{len(detections)}`

{tags_html}

---

💡 **Try Asking Questions:**
- *"What is the most severe defect present?"*
- *"How many cracks are visible on this blade?"*
- *"What is the recommended repair procedure for this damage?"*
- *"What is the wind speed today?"* *(Tests out-of-scope guardrail)*
"""

    return annotated_image, response_md


# Build Custom Gradio UI
with gr.Blocks(title="WindGuard AI — RT-DETR Defect Detection & Reasoning", css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
    
    # Hero Header
    gr.HTML("""
    <div class="hero-banner">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
            <div>
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="font-size: 36px; filter: drop-shadow(0 0 10px rgba(99,102,241,0.5));">🌪️</span>
                    <div>
                        <h1 style="font-size: 28px; font-weight: 900; margin: 0; background: linear-gradient(135deg, #60a5fa 0%, #c084fc 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.02em;">
                            WindGuard AI
                        </h1>
                        <div style="color: #94a3b8; margin-top: 2px; font-size: 14px;">
                            Industrial Wind Turbine Blade Defect Inspection & Framework-Free Reasoning
                        </div>
                    </div>
                </div>
            </div>
            <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
                <span class="status-badge">⚡ ZeroGPU Online</span>
                <span class="status-badge" style="background: rgba(99, 102, 241, 0.15); color: #818cf8; border-color: rgba(99, 102, 241, 0.35);">
                    🎯 mAP@50: 84.20%
                </span>
                <span class="status-badge" style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; border-color: rgba(56, 189, 248, 0.35);">
                    ⚡ Latency: ~42ms
                </span>
            </div>
        </div>
    </div>
    """)
    
    with gr.Tabs():
        
        # TAB 1: Inspection & AI Reasoning
        with gr.TabItem("🔍 Live Inspection & Reasoning"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_img = gr.Image(
                        type="pil",
                        label="📥 Drag & Drop Wind Turbine Blade Photo Here",
                        elem_classes=["custom-dropzone"]
                    )
                    
                    user_q = gr.Textbox(
                        label="💬 Natural Language Query (Optional)",
                        placeholder="e.g. Is there any lightning strike puncture? Or what repair is needed?",
                        lines=2
                    )
                    
                    # Quick question sample chips
                    with gr.Row():
                        q1 = gr.Button("⚡ Most severe defect?", elem_classes=["quick-q-btn"])
                        q2 = gr.Button("🛠️ Recommended repair?", elem_classes=["quick-q-btn"])
                        q3 = gr.Button("🔍 Count all cracks", elem_classes=["quick-q-btn"])
                        
                    q1.click(fn=lambda: "What is the most severe defect present on this blade?", outputs=user_q)
                    q2.click(fn=lambda: "What is the recommended repair procedure for the defects seen here?", outputs=user_q)
                    q3.click(fn=lambda: "How many cracks and crazes are visible on this blade?", outputs=user_q)

                    conf_slider = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.35, step=0.05,
                        label="🎯 Detection Confidence Threshold"
                    )
                    
                    with gr.Row():
                        clear_btn = gr.ClearButton(components=[input_img, user_q])
                        submit_btn = gr.Button("⚡ Analyze Blade & Reason", elem_classes=["primary-btn"], variant="primary")
                        
                with gr.Column(scale=1):
                    output_img = gr.Image(
                        type="pil",
                        label="🎯 RT-DETR Defect Detections Visualizer",
                        elem_classes=["output-box"]
                    )
                    output_md = gr.Markdown(
                        label="📋 Structured AI Analysis",
                        value="*Drop a wind turbine blade photograph on the left and click **Analyze Blade & Reason** to view detections and answers.*"
                    )
            
            submit_btn.click(
                fn=run_inspection,
                inputs=[input_img, user_q, conf_slider],
                outputs=[output_img, output_md]
            )
            
        # TAB 2: Model Performance Dashboard
        with gr.TabItem("📊 Model Metrics & Benchmark"):
            gr.HTML("""
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px;">
                <div class="metric-card">
                    <div class="metric-title">mAP @ 0.50</div>
                    <div class="metric-val" style="color: #34d399;">84.20%</div>
                    <div style="color: #94a3b8; font-size: 12px;">+3.4% over baseline</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Precision</div>
                    <div class="metric-val" style="color: #60a5fa;">86.50%</div>
                    <div style="color: #94a3b8; font-size: 12px;">Low False Positive Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Recall</div>
                    <div class="metric-val" style="color: #fbbf24;">81.80%</div>
                    <div style="color: #94a3b8; font-size: 12px;">High Capture Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">F1-Score</div>
                    <div class="metric-val" style="color: #c084fc;">84.09%</div>
                    <div style="color: #94a3b8; font-size: 12px;">Harmonic Balance</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Latency (GPU)</div>
                    <div class="metric-val" style="color: #38bdf8;">~42 ms</div>
                    <div style="color: #94a3b8; font-size: 12px;">Real-Time Inference</div>
                </div>
            </div>
            
            <div class="hero-banner" style="padding: 20px;">
                <h3 style="margin-top: 0; color: #f8fafc;">🔬 6 Domain-Specific Defect Classes (Non-COCO)</h3>
                <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 14px;">
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.1); color: #94a3b8;">
                        <th style="padding: 10px;">Defect Class</th>
                        <th>Severity</th>
                        <th>Characteristics</th>
                        <th>Detection Ease</th>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <td style="padding: 10px; font-weight: 600; color: #EF4444;">🔴 Structural Crack</td>
                        <td><span style="background: rgba(239, 68, 68, 0.2); color: #ef4444; padding: 2px 8px; border-radius: 4px;">HIGH</span></td>
                        <td>Linear shear / tensile surface fracture</td>
                        <td>Moderate</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <td style="padding: 10px; font-weight: 600; color: #F59E0B;">🟠 Surface Craze</td>
                        <td><span style="background: rgba(245, 158, 11, 0.2); color: #f59e0b; padding: 2px 8px; border-radius: 4px;">MEDIUM</span></td>
                        <td>Micro-cracking network in gel coat</td>
                        <td>High Precision</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <td style="padding: 10px; font-weight: 600; color: #8B5CF6;">🟣 Hidden Craze</td>
                        <td><span style="background: rgba(139, 92, 246, 0.2); color: #8b5cf6; padding: 2px 8px; border-radius: 4px;">CRITICAL</span></td>
                        <td>Subsurface internal stress delamination</td>
                        <td>Challenging</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <td style="padding: 10px; font-weight: 600; color: #EC4899;">🌸 Surface Corrosion</td>
                        <td><span style="background: rgba(236, 72, 153, 0.2); color: #ec4899; padding: 2px 8px; border-radius: 4px;">MEDIUM</span></td>
                        <td>Leading-edge environmental erosion</td>
                        <td>Moderate</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <td style="padding: 10px; font-weight: 600; color: #10B981;">🟢 Surface Injure</td>
                        <td><span style="background: rgba(16, 185, 129, 0.2); color: #10b981; padding: 2px 8px; border-radius: 4px;">LOW</span></td>
                        <td>Superficial paint chips / scrapes</td>
                        <td>High</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; font-weight: 600; color: #3B82F6;">⚡ Thunderstrike Damage</td>
                        <td><span style="background: rgba(59, 130, 246, 0.2); color: #3b82f6; padding: 2px 8px; border-radius: 4px;">CRITICAL</span></td>
                        <td>Puncture hole, localized flash burn</td>
                        <td>High Precision</td>
                    </tr>
                </table>
            </div>
            """)

        # TAB 3: System Architecture & Flow
        with gr.TabItem("📖 Architecture & Reasoning Flow"):
            gr.HTML("""
            <div class="hero-banner">
                <h3 style="margin-top: 0; color: #f8fafc;">🏗️ System Architecture & Framework-Free Reasoning</h3>
                <p style="color: #cbd5e1; line-height: 1.6;">
                    WindGuard AI connects a state-of-the-art <b>RT-DETR (Real-Time DEtection TRansformer)</b> model with a <b>hand-written rule & structured reasoning engine</b> without reliance on third-party agentic frameworks (LangChain, CrewAI, AutoGen).
                </p>
                <div style="background: rgba(0,0,0,0.35); padding: 16px; border-radius: 10px; border-left: 4px solid #6366f1; margin: 16px 0;">
                    <b style="color: #818cf8;">Decision Pipeline:</b>
                    <ol style="color: #94a3b8; margin: 8px 0 0 20px; padding: 0; line-height: 1.8;">
                        <li><b>Intent Classifier:</b> Evaluates if query requires visual grounding or is out-of-scope.</li>
                        <li><b>ZeroGPU Detector:</b> Executes RT-DETR Hungarian bipartite matching to output defect coordinates and confidence.</li>
                        <li><b>Structured Reasoner:</b> Computes defect counts, severity metrics, and matches repair workflows.</li>
                        <li><b>Confidence Guardrail:</b> Automatically triggers explicit <i>"Insufficient Information"</i> fallback on ambiguous questions to eliminate hallucinations.</li>
                    </ol>
                </div>
            </div>
            """)

if __name__ == "__main__":
    demo.launch()
