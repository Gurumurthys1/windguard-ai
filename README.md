# WindGuard AI — RT-DETR Defect Detection & Reasoning Platform

**Submission for RAP AI/ML Internship Assignment (Pre-Hackathon Screening Round 1)**  
*Domain*: Wind-Turbine Blade Defect (WTBD) Detection + Hand-Written AI Reasoning Layer  
*API Framework*: FastAPI + PyTorch + Hugging Face Transformers  
*Frontend UI*: Vite + React 19 + Lucide Icons + Recharts  

---

## 🌐 Live Demo

[![Live Demo on Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue?style=for-the-badge)](https://huggingface.co/spaces/Gurus01/windguard-ai)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Gurumurthys1%2Fwindguard--ai-black?style=for-the-badge&logo=github)](https://github.com/Gurumurthys1/windguard-ai)

> 🚀 **Try it now — no installation required:**  
> **[https://huggingface.co/spaces/Gurus01/windguard-ai](https://huggingface.co/spaces/Gurus01/windguard-ai)**  
> Upload a wind turbine blade image → Get RT-DETR defect bounding boxes + natural language AI reasoning instantly.

| | |
|---|---|
| **Live App** | [https://huggingface.co/spaces/Gurus01/windguard-ai](https://huggingface.co/spaces/Gurus01/windguard-ai) |
| **GitHub Repo** | [https://github.com/Gurumurthys1/windguard-ai](https://github.com/Gurumurthys1/windguard-ai) |
| **Hardware** | ZeroGPU (Free Tier) |
| **Model** | RT-DETR ResNet-50 Fine-Tuned |
| **mAP@50** | **84.20%** |
| **Inference Latency** | ~42 ms |

---

## 🚀 Quick Start & Web Application UI

### Option 1: Run via Docker (Recommended)
```bash
# 1. Build Docker image
docker build -t windguard-ai .

# 2. Run container
docker run -p 8000:8000 windguard-ai

# 3. Access Web UI Dashboard in Browser:
# http://localhost:8000
```

### Option 2: Run Locally (Python Virtual Environment)
```bash
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Start FastAPI Server
python -m uvicorn src.app:app --host 0.0.0.0 --port 8000

# 3. Access Dashboard at http://localhost:8000
```

---

## 📡 API Endpoints & Sample Payloads

### 1. `POST /detect` — Object Detection Endpoint
Uploads a turbine blade inspection photo and returns detected non-COCO defect classes (`delamination`, `crack`, `erosion`, `lightning_strike`, `scratch`), bounding box coordinates, and confidence scores.

#### Sample Request (cURL):
```bash
curl -X 'POST' \
  'http://localhost:8000/detect?score_threshold=0.30' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@data/raw/Images/1009.jpg'
```

#### Sample Response Payload:
```json
{
  "image_id": "1009.jpg",
  "device_used": "CPU",
  "detections": [
    {
      "class": "delamination",
      "confidence": 0.8942,
      "bbox": [142.5, 310.0, 512.0, 480.5]
    },
    {
      "class": "crack",
      "confidence": 0.7615,
      "bbox": [55.0, 120.2, 180.0, 240.0]
    }
  ],
  "count": 2,
  "latency_ms": 18.4
}
```

---

### 2. `POST /ask` — AI Reasoning Assistant Endpoint (Hand-Written No-Framework Logic)
Accepts a natural-language question + optional image and performs Intent Routing, Detection Execution, and Confidence Guardrails.

#### Sample Request (cURL):
```bash
curl -X 'POST' \
  'http://localhost:8000/ask' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'question=What repair procedure is required for this delamination?' \
  -F 'image=@data/raw/Images/1009.jpg'
```

#### Sample Response Payload (Structured Reasoning):
```json
{
  "intent": "detection_required",
  "detection_count": 2,
  "answer": "Diagnostic Findings:\n- Detected 1 delamination defect (Confidence: 89.4%) and 1 crack (Confidence: 76.2%).\n\nRecommended Action:\nDelamination represents critical composite layer separation. Immediate blade shutdown is recommended followed by vacuum-assisted resin injection (VARI) repair within 48 hours."
}
```

#### Sample Response Payload (Confidence Guardrail - Insufficient Information):
```json
{
  "intent": "detection_required",
  "detection_count": 0,
  "answer": "Insufficient Information: The model detected no defects above the required confidence threshold (0.30) to answer this question with certainty."
}
```

---

## 📊 Evaluation & Model Accuracy

| Metric | Score (%) | Technical Notes |
| :--- | :---: | :--- |
| **mAP@50 (IoU = 0.50)** | **`84.20%`** | Detection accuracy across non-COCO defect classes |
| **Precision** | **`86.50%`** | Proportion of flagged defects that are true positives |
| **Recall** | **`81.80%`** | Sensitivity / Percentage of actual defects found |
| **F1-Score** | **`84.09%`** | Harmonic balance between Precision & Recall |
| **Error Reduction** | **`94.38%`** | Training loss decreased from **227.89** to **12.80** |

Detailed 2-page Written Memo & 5 Failure Case Analyses are documented in [`docs/written_memo.md`](file:///c:/Users/admin/Downloads/ml_intern-task/windguard-ai/docs/written_memo.md).

---

## 📂 Repository Structure

```
windguard-ai/
├── README.md                           # Documentation & Run instructions
├── Dockerfile                          # Production Docker container setup
├── requirements.txt                    # Python dependencies
├── src/                                # FastAPI Backend API
│   ├── app.py                          # Main FastAPI server & static routes
│   ├── detector.py                     # RT-DETR model wrapper
│   ├── reasoning.py                    # Hand-written decision & reasoning layer
│   ├── routes_detect.py                # POST /detect route
│   └── routes_ask.py                   # POST /ask route
├── frontend/                           # React Web UI Dashboard
│   ├── src/                            # App.jsx, index.css, assets
│   ├── package.json                    # React dependencies
│   └── vite.config.js                  # Vite configuration & proxy
├── models/                             # Model checkpoints & final weights
│   └── rtdetr-wtbd-final/             # Fine-tuned model weights (safetensors)
├── data/                               # WTBD dataset (train/val/test splits)
└── docs/                               # Written Memo & evaluation loss plots
    ├── written_memo.md                 # 2-Page RAP Submission Technical Memo
    └── training_loss_plot.png          # Loss progression graph
```
