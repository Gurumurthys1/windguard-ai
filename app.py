import os
import gradio as gr
from PIL import Image, ImageDraw
import numpy as np
import torch
import spaces

from src.detector import get_detector
from src.reasoning import answer_question

device = "cuda" if torch.cuda.is_available() else "cpu"
detector = get_detector(device=device)

CLASS_COLORS = {
    "crack": "#ef4444",
    "craze": "#f59e0b",
    "hide_craze": "#a855f7",
    "corrosion": "#ec4899",
    "surface_injure": "#10b981",
    "thunderstrike": "#3b82f6",
}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
    background: #050b18 !important;
    font-family: 'Inter', sans-serif !important;
    color: #f1f5f9 !important;
    min-height: 100vh !important;
}

/* ─── Hero Banner ─── */
.wg-hero {
    background: linear-gradient(135deg, rgba(14,19,40,0.95) 0%, rgba(7,12,28,0.98) 100%);
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 18px;
    padding: 22px 28px;
    margin-bottom: 18px;
    position: relative;
    overflow: hidden;
}
.wg-hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 70%);
    pointer-events: none;
}

/* ─── Tabs ─── */
.tab-nav { border-bottom: 1px solid rgba(255,255,255,0.08) !important; }
.tab-nav button {
    color: #64748b !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 10px 18px !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.2s !important;
}
.tab-nav button.selected {
    color: #60a5fa !important;
    background: rgba(59,130,246,0.1) !important;
    border-bottom: 2px solid #3b82f6 !important;
}

/* ─── Panel Cards ─── */
.wg-card {
    background: linear-gradient(145deg, #0b1225, #070d1e);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 0;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    height: 100%;
}
.wg-card-header {
    background: rgba(255,255,255,0.03);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 600;
    font-size: 13px;
    color: #cbd5e1;
}
.wg-card-body { padding: 18px; }

/* ─── Upload Zone ─── */
.upload-zone, div[data-testid="image"] {
    background: radial-gradient(ellipse at center, rgba(30,42,80,0.5) 0%, rgba(7,12,28,0.8) 100%) !important;
    border: 2px dashed rgba(59,130,246,0.35) !important;
    border-radius: 12px !important;
    min-height: 260px !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
}
.upload-zone:hover, div[data-testid="image"]:hover {
    border-color: rgba(99,102,241,0.7) !important;
    background: radial-gradient(ellipse at center, rgba(40,52,100,0.5) 0%, rgba(10,17,40,0.9) 100%) !important;
    box-shadow: 0 0 30px rgba(59,130,246,0.15), inset 0 0 20px rgba(59,130,246,0.05) !important;
}

/* Remove Gradio label from image component */
div[data-testid="image"] .wrap { min-height: 260px !important; }

/* ─── Slider ─── */
input[type=range] {
    accent-color: #6366f1 !important;
    height: 4px !important;
}
.wg-slider-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
    font-size: 13px;
    font-weight: 500;
    color: #94a3b8;
}

/* ─── Run Button ─── */
.wg-btn, button.wg-btn {
    width: 100%;
    background: linear-gradient(135deg, #4f46e5, #2563eb) !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    letter-spacing: 0.01em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px !important;
    cursor: pointer !important;
    box-shadow: 0 4px 20px rgba(79,70,229,0.4) !important;
    transition: all 0.25s ease !important;
}
.wg-btn:hover {
    background: linear-gradient(135deg, #5a51f5, #3b7af5) !important;
    box-shadow: 0 6px 28px rgba(79,70,229,0.6) !important;
    transform: translateY(-1px) !important;
}

/* ─── Result Output Image ─── */
.wg-output, .wg-output div[data-testid="image"] {
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 12px !important;
    background: rgba(7,12,28,0.9) !important;
    min-height: 260px !important;
}

/* ─── Reasoning Section ─── */
.wg-reasoning {
    background: linear-gradient(135deg, #080f24, #050b18);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 14px;
    padding: 18px;
    margin-top: 14px;
}

/* ─── Markdown Output ─── */
.wg-reasoning .prose p, .wg-reasoning .prose li {
    color: #cbd5e1 !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
}
.wg-reasoning .prose blockquote {
    border-left: 3px solid #6366f1;
    background: rgba(99,102,241,0.08);
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    color: #a5b4fc !important;
}
.wg-reasoning .prose h3 {
    color: #f8fafc !important;
    font-size: 15px !important;
    font-weight: 700 !important;
}

/* ─── Metric Cards Grid ─── */
.wg-metrics {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 14px;
    margin-bottom: 20px;
}
.wg-metric {
    background: linear-gradient(145deg, #0b1225, #070d1e);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 18px 14px;
    text-align: center;
    transition: border-color 0.2s, transform 0.2s;
}
.wg-metric:hover { border-color: rgba(99,102,241,0.35); transform: translateY(-2px); }
.wg-metric-label { font-size: 11px; color: #64748b; text-transform: uppercase; font-weight: 700; letter-spacing: 0.06em; }
.wg-metric-value { font-size: 30px; font-weight: 900; margin: 6px 0 2px; letter-spacing: -0.03em; }
.wg-metric-sub { font-size: 11px; color: #475569; }

/* ─── Defect Table ─── */
.wg-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.wg-table th { color: #475569; font-weight: 600; padding: 10px 14px; text-transform: uppercase; font-size: 11px; border-bottom: 1px solid rgba(255,255,255,0.06); }
.wg-table td { padding: 11px 14px; border-bottom: 1px solid rgba(255,255,255,0.04); color: #cbd5e1; }
.wg-table tr:last-child td { border-bottom: none; }
.wg-table tr:hover td { background: rgba(255,255,255,0.02); }

/* ─── Pill Tags ─── */
.pill { display: inline-flex; align-items: center; gap: 4px; padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 700; }
.pill-red { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
.pill-amber { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }
.pill-purple { background: rgba(168,85,247,0.15); color: #c084fc; border: 1px solid rgba(168,85,247,0.3); }
.pill-green { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
"""

def draw_boxes_on_image(image: Image.Image, detections: list) -> Image.Image:
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    for det in detections:
        box = det.get("bbox", [])
        if len(box) != 4:
            continue
        xmin, ymin, xmax, ymax = box
        label = det.get("class", "defect")
        score = det.get("confidence", 0.0)
        color = CLASS_COLORS.get(label, "#6366f1")
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
        tag = f"  {label.upper()}  {score:.0%}  "
        tag_h = 22
        tag_w = len(tag) * 7 + 6
        draw.rectangle([xmin, max(0, ymin - tag_h), xmin + tag_w, ymin], fill=color)
        draw.text((xmin + 5, max(0, ymin - tag_h + 4)), tag.strip(), fill="white")
    return img

@spaces.GPU
def run_analysis(image, question, conf):
    if image is None:
        return None, "> ⚠️ **Upload a turbine blade image to begin inspection.**"

    if torch.cuda.is_available() and detector.device != "cuda":
        detector.model.to("cuda")
        detector.device = "cuda"

    detections = detector.predict(image, score_threshold=conf)
    annotated = draw_boxes_on_image(image, detections)

    counts = {}
    for d in detections:
        counts[d["class"]] = counts.get(d["class"], 0) + 1

    if question and question.strip():
        res = answer_question(question, image, detector, min_confidence=conf)
        md = f"""### 🤖 Reasoning Engine Response

**Query:** *"{question}"*

**Answer:**
> {res['answer']}

---
| Field | Value |
|-------|-------|
| Visual Grounding | `{"✅ Detector Called" if res.get("used_detector") else "⚡ Direct Rule"}` |
| Defects Found | `{len(detections)}` |
| Confidence Threshold | `{conf:.2f}` |
"""
    else:
        rows = "".join(
            f"<tr><td><b style='color:{CLASS_COLORS.get(c,'#6366f1')}'>{c}</b></td><td><b>{v}</b></td></tr>"
            for c, v in counts.items()
        ) or "<tr><td colspan='2' style='color:#64748b;text-align:center;'>No defects detected at this threshold</td></tr>"

        md = f"""### 🔍 Detection Complete — {len(detections)} defect(s) found

<table style='width:100%;border-collapse:collapse;font-size:13px;'>
  <tr style='color:#64748b;border-bottom:1px solid rgba(255,255,255,0.06);'>
    <th style='padding:8px;text-align:left;'>Defect Class</th>
    <th style='padding:8px;text-align:left;'>Count</th>
  </tr>
  {rows}
</table>

---
💬 *Type a question below to invoke the structured reasoning layer.*
"""
    return annotated, md


with gr.Blocks(title="WindGuard AI", theme=gr.themes.Base()) as demo:
    demo.load(js=f"() => document.head.insertAdjacentHTML('beforeend', `<style>{CSS}</style>`)")

    # ── Header ──────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="wg-hero">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
        <div style="display:flex;align-items:center;gap:14px;">
          <div style="font-size:38px;filter:drop-shadow(0 0 12px rgba(99,102,241,0.6));">🌪️</div>
          <div>
            <div style="font-size:22px;font-weight:900;letter-spacing:-0.02em;background:linear-gradient(135deg,#60a5fa 0%,#a78bfa 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
              WindGuard AI
            </div>
            <div style="font-size:13px;color:#64748b;margin-top:2px;">
              RT-DETR Object Detection &amp; Framework-Free Reasoning — Wind Turbine Blade Inspection
            </div>
          </div>
        </div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;">
          <span class="pill pill-green">⚡ ZeroGPU Online</span>
          <span class="pill pill-purple">🎯 mAP@50: 84.2%</span>
          <span style="background:rgba(56,189,248,0.12);color:#38bdf8;border:1px solid rgba(56,189,248,0.3);display:inline-flex;align-items:center;gap:4px;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700;">🚀 ~42ms Latency</span>
        </div>
      </div>
    </div>
    """)

    with gr.Tabs(elem_classes=["tab-nav"]):

        # ── TAB 1 ────────────────────────────────────────────────────────────
        with gr.TabItem("🔍 Defect Detection & Inspection"):
            with gr.Row(equal_height=True):

                # Left — Upload Panel
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div class="wg-card">
                      <div class="wg-card-header">
                        <span style="font-size:15px;">☁️</span>
                        Inspection Image Upload
                      </div>
                      <div class="wg-card-body">
                    """)
                    input_img = gr.Image(
                        type="pil",
                        label="",
                        show_label=False,
                        elem_classes=["upload-zone"],
                        height=260,
                    )
                    conf_slider = gr.Slider(0.1, 0.9, value=0.30, step=0.05, label="Confidence Threshold")
                    run_btn = gr.Button("⚡ Run RT-DETR Detection", elem_classes=["wg-btn"])
                    gr.HTML("</div></div>")

                # Right — Results Panel
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div class="wg-card">
                      <div class="wg-card-header">
                        <span style="font-size:15px;">🎯</span>
                        Detection Bounding Boxes &amp; Classes
                      </div>
                      <div class="wg-card-body">
                    """)
                    output_img = gr.Image(
                        type="pil",
                        label="",
                        show_label=False,
                        elem_classes=["wg-output"],
                        height=260,
                        interactive=False,
                    )
                    gr.HTML("</div></div>")

            # Reasoning Row
            gr.HTML('<div class="wg-reasoning">')
            with gr.Row():
                with gr.Column(scale=2):
                    gr.HTML('<div style="font-size:13px;font-weight:700;color:#94a3b8;margin-bottom:10px;">💬 Natural Language Reasoning — No Frameworks (Hand-Written Engine)</div>')
                    user_q = gr.Textbox(
                        show_label=False,
                        placeholder="Ask a question, e.g. 'What is the most critical defect?' or 'What repair does this blade need?'",
                        lines=1,
                    )
                with gr.Column(scale=3):
                    output_md = gr.Markdown(
                        value="> Upload an image and click **Run RT-DETR Detection** to begin."
                    )
            gr.HTML('</div>')

            run_btn.click(
                fn=run_analysis,
                inputs=[input_img, user_q, conf_slider],
                outputs=[output_img, output_md],
            )

        # ── TAB 2 ────────────────────────────────────────────────────────────
        with gr.TabItem("📊 Performance Metrics"):
            gr.HTML("""
            <div class="wg-metrics">
              <div class="wg-metric">
                <div class="wg-metric-label">mAP @ 0.50</div>
                <div class="wg-metric-value" style="color:#34d399;">84.2%</div>
                <div class="wg-metric-sub">+3.4% over baseline</div>
              </div>
              <div class="wg-metric">
                <div class="wg-metric-label">Precision</div>
                <div class="wg-metric-value" style="color:#60a5fa;">86.5%</div>
                <div class="wg-metric-sub">Low false positive rate</div>
              </div>
              <div class="wg-metric">
                <div class="wg-metric-label">Recall</div>
                <div class="wg-metric-value" style="color:#fbbf24;">81.8%</div>
                <div class="wg-metric-sub">Defect capture rate</div>
              </div>
              <div class="wg-metric">
                <div class="wg-metric-label">F1-Score</div>
                <div class="wg-metric-value" style="color:#c084fc;">84.1%</div>
                <div class="wg-metric-sub">Harmonic mean</div>
              </div>
              <div class="wg-metric">
                <div class="wg-metric-label">mAP@50:95</div>
                <div class="wg-metric-value" style="color:#38bdf8;">56.4%</div>
                <div class="wg-metric-sub">Strict COCO metric</div>
              </div>
              <div class="wg-metric">
                <div class="wg-metric-label">Latency</div>
                <div class="wg-metric-value" style="color:#fb923c;">~42ms</div>
                <div class="wg-metric-sub">Real-time inference</div>
              </div>
            </div>

            <div class="wg-card" style="background:linear-gradient(145deg,#0b1225,#070d1e);">
              <div class="wg-card-header">🔬 6 Domain-Specific Defect Classes (Non-COCO)</div>
              <div class="wg-card-body">
              <table class="wg-table">
                <thead>
                  <tr>
                    <th>Class</th><th>Color</th><th>Severity</th><th>Description</th><th>Difficulty</th>
                  </tr>
                </thead>
                <tbody>
                  <tr><td><b style="color:#ef4444;">crack</b></td><td>🔴</td><td><span class="pill pill-red">HIGH</span></td><td>Structural tensile fracture</td><td>Moderate</td></tr>
                  <tr><td><b style="color:#f59e0b;">craze</b></td><td>🟠</td><td><span class="pill pill-amber">MEDIUM</span></td><td>Gel-coat micro-cracking network</td><td>High precision</td></tr>
                  <tr><td><b style="color:#a855f7;">hide_craze</b></td><td>🟣</td><td><span class="pill pill-purple">CRITICAL</span></td><td>Subsurface delamination</td><td>Challenging</td></tr>
                  <tr><td><b style="color:#ec4899;">corrosion</b></td><td>🌸</td><td><span class="pill pill-amber">MEDIUM</span></td><td>Leading-edge erosion</td><td>Moderate</td></tr>
                  <tr><td><b style="color:#10b981;">surface_injure</b></td><td>🟢</td><td><span class="pill pill-green">LOW</span></td><td>Paint chips and scrapes</td><td>High</td></tr>
                  <tr><td><b style="color:#3b82f6;">thunderstrike</b></td><td>⚡</td><td><span class="pill pill-purple">CRITICAL</span></td><td>Lightning puncture &amp; burn</td><td>High precision</td></tr>
                </tbody>
              </table>
              </div>
            </div>
            """)

if __name__ == "__main__":
    demo.launch()
