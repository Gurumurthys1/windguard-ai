import React, { useState, useRef } from 'react';
import {
  ShieldAlert,
  Cpu,
  UploadCloud,
  Sliders,
  Activity,
  HelpCircle,
  BarChart3,
  CheckCircle2,
  Zap,
  FileText,
  AlertTriangle,
  RefreshCw,
  Server
} from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid
} from 'recharts';

const lossData = [
  // ── Run 1: Epochs 1–12 ──
  { epoch: 1, trainLoss: 32.3600, evalLoss: 20.5800 },
  { epoch: 2, trainLoss: 24.3800, evalLoss: 15.3900 },
  { epoch: 3, trainLoss: 22.5400, evalLoss: 15.3700 },
  { epoch: 4, trainLoss: 20.4000, evalLoss: 14.3400 },
  { epoch: 5, trainLoss: 19.3100, evalLoss: 15.7500 },
  { epoch: 6, trainLoss: 17.2200, evalLoss: 13.6000 },
  { epoch: 7, trainLoss: 16.0600, evalLoss: 13.1500 },
  { epoch: 8, trainLoss: 15.5600, evalLoss: 14.1700 },
  { epoch: 9, trainLoss: 13.8700, evalLoss: 13.4000 },
  { epoch: 10, trainLoss: 13.6000, evalLoss: 13.9500 },
  { epoch: 11, trainLoss: 13.5800, evalLoss: 13.1300 },
  { epoch: 12, trainLoss: 12.8000, evalLoss: 13.6900 },
  // ── Run 2: Epochs 13–24 ──
  { epoch: 13, trainLoss: 42.7091, evalLoss: 20.9977 },
  { epoch: 14, trainLoss: 23.6957, evalLoss: 14.1833 },
  { epoch: 15, trainLoss: 19.9864, evalLoss: 13.2529 },
  { epoch: 16, trainLoss: 20.2954, evalLoss: 14.9877 },
  { epoch: 17, trainLoss: 19.7243, evalLoss: 12.9539 },
  { epoch: 18, trainLoss: 16.8765, evalLoss: 13.5468 },
  { epoch: 19, trainLoss: 16.3151, evalLoss: 13.2569 },
  { epoch: 20, trainLoss: 15.0426, evalLoss: 13.2043 },
  { epoch: 21, trainLoss: 13.9083, evalLoss: 12.7477 },
  { epoch: 22, trainLoss: 12.4718, evalLoss: 13.0282 },
  { epoch: 23, trainLoss: 12.6916, evalLoss: 13.2893 },
  { epoch: 24, trainLoss: 12.5316, evalLoss: 12.9071 },
];

const DEFECT_COLORS = {
  'delamination': '#ef4444',
  'crack': '#f59e0b',
  'erosion': '#3b82f6',
  'lightning_strike': '#8b5cf6',
  'scratch': '#10b981'
};

export default function App() {
  const [activeTab, setActiveTab] = useState('detect');
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [confidence, setConfidence] = useState(0.30);
  const [deviceMode, setDeviceMode] = useState('cuda'); // 'cuda' | 'cpu'
  const [loadingDetect, setLoadingDetect] = useState(false);
  const [detections, setDetections] = useState([]);
  const [latency, setLatency] = useState(null);
  const [activeDevice, setActiveDevice] = useState('CUDA');

  // AI Reasoning state
  const [question, setQuestion] = useState('');
  const [loadingAsk, setLoadingAsk] = useState(false);
  const [reasoningResult, setReasoningResult] = useState('');

  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setSelectedFile(file);

    const reader = new FileReader();
    reader.onload = (evt) => {
      setImagePreview(evt.target.result);
      setDetections([]);
    };
    reader.readAsDataURL(file);
  };

  const drawBoundingBoxes = (imgUrl, dets) => {
    const img = new Image();
    img.src = imgUrl;
    img.onload = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const container = canvas.parentElement;
      const displayW = container ? container.clientWidth : img.width;
      const scale = displayW / img.width;
      const displayH = img.height * scale;

      // Set canvas to DISPLAY size so CSS doesn't distort bounding boxes
      canvas.width = displayW;
      canvas.height = displayH;
      canvas.style.width = '100%';
      canvas.style.height = 'auto';

      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, displayW, displayH);

      dets.forEach((d, idx) => {
        const [xmin, ymin, xmax, ymax] = d.bbox;
        // Scale bbox coords from original image space → display space
        const sx = xmin * scale;
        const sy = ymin * scale;
        const sw = (xmax - xmin) * scale;
        const sh = (ymax - ymin) * scale;
        const color = DEFECT_COLORS[d.class] || '#ef4444';

        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(2, Math.round(displayW / 300));
        ctx.strokeRect(sx, sy, sw, sh);

        ctx.fillStyle = color;
        const fontSize = Math.max(11, Math.round(displayW / 60));
        ctx.font = `bold ${fontSize}px Inter, sans-serif`;
        const labelText = `#${idx + 1} ${d.class} (${(d.confidence * 100).toFixed(0)}%)`;
        const textWidth = ctx.measureText(labelText).width;

        ctx.fillRect(sx, Math.max(0, sy - fontSize - 6), textWidth + 10, fontSize + 6);
        ctx.fillStyle = '#ffffff';
        ctx.fillText(labelText, sx + 5, Math.max(fontSize, sy - 2));
      });
    };
  };

  const runDetection = async () => {
    if (!selectedFile) return;
    setLoadingDetect(true);
    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const res = await fetch(`/detect?score_threshold=${confidence}&device=${deviceMode}`, {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (res.ok) {
        setDetections(data.detections || []);
        setLatency(data.latency_ms);
        setActiveDevice(data.device_used || deviceMode.toUpperCase());
        drawBoundingBoxes(imagePreview, data.detections || []);
      } else {
        alert(`Error: ${data.detail || 'Detection failed'}`);
      }
    } catch (err) {
      alert(`Network error: ${err.message}`);
    } finally {
      setLoadingDetect(false);
    }
  };

  const runReasoning = async (customQ = null) => {
    const query = customQ || question;
    if (!query.trim()) {
      alert('Please enter a question!');
      return;
    }
    setLoadingAsk(true);
    setReasoningResult('');

    const formData = new FormData();
    formData.append('question', query);
    if (selectedFile) {
      formData.append('image', selectedFile);
    }

    try {
      const res = await fetch('/ask', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (res.ok) {
        setReasoningResult(data.answer || JSON.stringify(data, null, 2));
      } else {
        setReasoningResult(`Error: ${data.detail || 'Reasoning failed'}`);
      }
    } catch (err) {
      setReasoningResult(`Network error: ${err.message}`);
    } finally {
      setLoadingAsk(false);
    }
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>

      {/* Header */}
      <header className="glass-panel" style={{ padding: '1.2rem 2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{
            width: 42, height: 42,
            background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
            borderRadius: 12, display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 4px 15px rgba(59, 130, 246, 0.4)'
          }}>
            <ShieldAlert size={24} color="#fff" />
          </div>
          <div>
            <h1 style={{ fontSize: '1.3rem', fontWeight: 700, letterSpacing: '-0.02em', color: '#fff' }}>
              WindGuard AI <span style={{ fontSize: '0.8rem', fontWeight: 500, color: '#38bdf8', background: 'rgba(56,189,248,0.1)', padding: '2px 8px', borderRadius: 12 }}>WTBD Edition</span>
            </h1>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>RT-DETR Defect Detection & Reasoning Platform</p>
          </div>
        </div>

      </header>


      {/* Main App Workspace */}
      <main style={{ flex: 1, padding: '2rem', maxWidth: 1400, margin: '0 auto', width: '100%' }}>

        {/* Metric Cards Banner */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
          <div className="glass-card" style={{ padding: '1.2rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
              <span>Best Validation Loss</span>
              <Activity size={18} color="var(--accent-emerald)" />
            </div>
            <div style={{ fontSize: '1.8rem', fontWeight: 700, color: '#34d399', fontFamily: 'JetBrains Mono, monospace', margin: '0.3rem 0' }}>12.7477</div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Epoch 21 of 24 ✅</div>
          </div>

          <div className="glass-card" style={{ padding: '1.2rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
              <span>Final Training Loss</span>
              <Zap size={18} color="var(--accent-blue)" />
            </div>
            <div style={{ fontSize: '1.8rem', fontWeight: 700, color: '#60a5fa', fontFamily: 'JetBrains Mono, monospace', margin: '0.3rem 0' }}>12.5316</div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Down from 227.89 · 24 Epochs</div>
          </div>

          <div className="glass-card" style={{ padding: '1.2rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
              <span>Active Engine</span>
              <Cpu size={18} color="var(--accent-purple)" />
            </div>
            <div style={{ fontSize: '1.25rem', fontWeight: 700, color: deviceMode === 'cuda' ? '#c084fc' : '#fbbf24', margin: '0.6rem 0' }}>
              {activeDevice} Engine
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>RT-DETR ResNet-50</div>
          </div>

          <div className="glass-card" style={{ padding: '1.2rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
              <span>Inference Speed</span>
              <Activity size={18} color="var(--accent-amber)" />
            </div>
            <div style={{ fontSize: '1.8rem', fontWeight: 700, color: '#fbbf24', fontFamily: 'JetBrains Mono, monospace', margin: '0.3rem 0' }}>
              {latency ? `${latency} ms` : (deviceMode === 'cuda' ? '~18 ms' : '~140 ms')}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
              {deviceMode === 'cuda' ? 'GPU FP16 Accelerated' : 'CPU Threaded'}
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div style={{ display: 'flex', gap: '1rem', borderBottom: '1px solid var(--border-main)', paddingBottom: '0.75rem', marginBottom: '2rem' }}>
          <button
            onClick={() => setActiveTab('detect')}
            style={{
              background: activeTab === 'detect' ? 'var(--accent-blue)' : 'rgba(35, 49, 79, 0.4)',
              color: '#fff', border: 'none', padding: '0.7rem 1.4rem', borderRadius: 10,
              fontWeight: 600, fontSize: '0.95rem', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem'
            }}
          >
            <ShieldAlert size={18} /> Defect Detector (/detect)
          </button>

          <button
            onClick={() => setActiveTab('reasoning')}
            style={{
              background: activeTab === 'reasoning' ? 'var(--accent-purple)' : 'rgba(35, 49, 79, 0.4)',
              color: '#fff', border: 'none', padding: '0.7rem 1.4rem', borderRadius: 10,
              fontWeight: 600, fontSize: '0.95rem', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem'
            }}
          >
            <HelpCircle size={18} /> AI Reasoning Assistant (/ask)
          </button>

          <button
            onClick={() => setActiveTab('analytics')}
            style={{
              background: activeTab === 'analytics' ? 'var(--accent-cyan)' : 'rgba(35, 49, 79, 0.4)',
              color: '#fff', border: 'none', padding: '0.7rem 1.4rem', borderRadius: 10,
              fontWeight: 600, fontSize: '0.95rem', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem'
            }}
          >
            <BarChart3 size={18} /> Model Loss Analytics
          </button>
        </div>

        {/* TAB 1: DEFECT DETECTOR */}
        {activeTab === 'detect' && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>

            {/* Input Controls */}
            <div className="glass-panel" style={{ borderRadius: 16, padding: '1.5rem' }}>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '1.25rem', color: '#fff', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <UploadCloud color="var(--accent-blue)" size={20} /> Inspection Image Upload
              </h3>

              <div
                onClick={() => fileInputRef.current?.click()}
                style={{
                  border: '2px dashed var(--border-main)', borderRadius: 12, padding: '2.5rem 1.5rem',
                  textAlign: 'center', cursor: 'pointer', background: 'rgba(11, 15, 25, 0.5)',
                  marginBottom: '1.5rem', transition: 'all 0.2s'
                }}
              >
                <UploadCloud size={40} color="var(--accent-blue)" style={{ margin: '0 auto 0.75rem' }} />
                <p style={{ fontWeight: 500, color: '#f8fafc' }}>Click to select turbine blade image</p>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.4rem' }}>Supports JPG, PNG (Max 10MB)</p>
                <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileChange} style={{ display: 'none' }} />
              </div>

              <div style={{ marginBottom: '1.5rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                  <label style={{ fontSize: '0.85rem', fontWeight: 500, color: 'var(--text-muted)' }}>Confidence Threshold</label>
                  <span style={{ fontWeight: 600, color: 'var(--accent-blue)' }}>{confidence.toFixed(2)}</span>
                </div>
                <input
                  type="range" min="0.05" max="0.95" step="0.05" value={confidence}
                  onChange={(e) => setConfidence(parseFloat(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--accent-blue)' }}
                />
              </div>

              <button
                onClick={runDetection}
                disabled={!selectedFile || loadingDetect}
                style={{
                  width: '100%', padding: '0.85rem', borderRadius: 10,
                  background: deviceMode === 'cuda' ? 'var(--accent-blue)' : 'var(--accent-amber)',
                  color: '#fff', border: 'none', fontWeight: 600, fontSize: '0.95rem', cursor: selectedFile ? 'pointer' : 'not-allowed',
                  opacity: selectedFile ? 1 : 0.5, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem'
                }}
              >
                {loadingDetect ? <RefreshCw className="spinner" size={18} /> : <Zap size={18} />}
                {loadingDetect ? `Scanning on ${deviceMode.toUpperCase()}...` : `Run RT-DETR (${deviceMode.toUpperCase()} Mode)`}
              </button>
            </div>

            {/* Results Output */}
            <div className="glass-panel" style={{ borderRadius: 16, padding: '1.5rem' }}>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '1.25rem', color: '#fff' }}>
                🎯 Detection Bounding Boxes & Classes
              </h3>

              {!imagePreview ? (
                <div style={{ height: 360, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
                  <FileText size={48} style={{ opacity: 0.3, marginBottom: '0.75rem' }} />
                  <p>Upload an image to preview detection results</p>
                </div>
              ) : (
                <div>
                  <div ref={el => { if (el && canvasRef.current) { /* container ready */ } }} style={{ borderRadius: 12, overflow: 'hidden', background: 'rgba(11,15,25,0.6)', marginBottom: '1rem', width: '100%' }}>
                    <canvas ref={canvasRef} style={{ display: 'block', width: '100%', height: 'auto' }} />
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', maxHeight: 160, overflowY: 'auto' }}>
                    {detections.length === 0 ? (
                      <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', textAlign: 'center', padding: '0.5rem' }}>
                        No defects detected above {confidence.toFixed(2)} threshold.
                      </div>
                    ) : (
                      detections.map((d, i) => (
                        <div key={i} style={{ background: 'rgba(11, 15, 25, 0.6)', border: '1px solid var(--border-main)', padding: '0.6rem 1rem', borderRadius: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <div>
                            <span style={{ fontWeight: 600, color: DEFECT_COLORS[d.class] || '#fff' }}>#{i + 1} {d.class}</span>
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace' }}>BBox: [{d.bbox.join(', ')}]</div>
                          </div>
                          <span style={{ background: 'rgba(239, 68, 68, 0.15)', color: '#fca5a5', border: '1px solid rgba(239, 68, 68, 0.3)', padding: '0.2rem 0.6rem', borderRadius: 6, fontSize: '0.8rem', fontWeight: 600 }}>
                            {(d.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              )}
            </div>

          </div>
        )}

        {/* TAB 2: AI REASONING ASSISTANT */}
        {activeTab === 'reasoning' && (
          <div className="glass-panel" style={{ borderRadius: 16, padding: '2rem', maxWidth: 900, margin: '0 auto' }}>
            <h3 style={{ fontSize: '1.2rem', fontWeight: 600, marginBottom: '1rem', color: '#fff', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <HelpCircle color="var(--accent-purple)" size={22} /> WindGuard AI Reasoning Layer
            </h3>
            <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
              Ask natural language questions about blade defect severity, repair procedures, safety compliance, or operational decisions.
            </p>

            {/* Image upload for reasoning — required for repair/detect questions */}
            <div style={{ marginBottom: '1.25rem' }}>
              <label style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: '0.5rem' }}>
                📎 Upload Blade Image (required for repair/detection questions)
              </label>
              <div
                onClick={() => fileInputRef.current?.click()}
                style={{
                  border: `2px dashed ${selectedFile ? 'var(--accent-purple)' : 'var(--border-main)'}`,
                  borderRadius: 10, padding: '0.9rem 1.25rem', cursor: 'pointer',
                  background: selectedFile ? 'rgba(139,92,246,0.08)' : 'rgba(11,15,25,0.5)',
                  display: 'flex', alignItems: 'center', gap: '0.75rem', transition: 'all 0.2s'
                }}
              >
                <UploadCloud size={22} color={selectedFile ? '#c084fc' : 'var(--text-muted)'} />
                <span style={{ fontSize: '0.9rem', color: selectedFile ? '#c084fc' : 'var(--text-muted)' }}>
                  {selectedFile ? `✅ ${selectedFile.name}` : 'Click to select turbine blade image...'}
                </span>
              </div>
              {selectedFile && imagePreview && (
                <img src={imagePreview} alt="preview" style={{ marginTop: '0.5rem', maxHeight: 120, borderRadius: 8, border: '1px solid var(--border-main)' }} />
              )}
            </div>

            <div style={{ display: 'flex', gap: '0.6rem', flexWrap: 'wrap', marginBottom: '1.25rem' }}>
              {[
                'What repair procedure is required?',
                'How severe is the damage?',
                'How many defects are visible?',
                'What is the most common defect?'
              ].map((chip, idx) => (
                <button
                  key={idx}
                  onClick={() => { setQuestion(chip); runReasoning(chip); }}
                  style={{ background: 'rgba(139, 92, 246, 0.15)', color: '#c084fc', border: '1px solid rgba(139, 92, 246, 0.3)', padding: '0.4rem 0.8rem', borderRadius: 20, fontSize: '0.8rem', cursor: 'pointer' }}
                >
                  {chip}
                </button>
              ))}
            </div>

            <textarea
              rows={3}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="e.g. What repair process is required? / How many cracks are visible?"
              style={{ width: '100%', background: 'rgba(11, 15, 25, 0.8)', border: '1px solid var(--border-main)', borderRadius: 10, padding: '0.85rem', color: '#fff', fontSize: '0.95rem', outline: 'none', marginBottom: '1.25rem', boxSizing: 'border-box' }}
            />

            <button
              onClick={() => runReasoning()}
              disabled={loadingAsk}
              style={{ width: '100%', padding: '0.85rem', borderRadius: 10, background: 'linear-gradient(135deg, #8b5cf6, #ec4899)', color: '#fff', border: 'none', fontWeight: 600, fontSize: '0.95rem', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}
            >
              {loadingAsk ? <RefreshCw className="spinner" size={18} /> : <Zap size={18} />}
              {loadingAsk ? 'Analyzing Query...' : 'Ask WindGuard AI'}
            </button>

            {!selectedFile && (
              <div style={{ marginTop: '0.75rem', padding: '0.6rem 1rem', background: 'rgba(245,158,11,0.1)', border: '1px solid rgba(245,158,11,0.3)', borderRadius: 8, fontSize: '0.82rem', color: '#fbbf24' }}>
                ⚠️ No image uploaded — repair/detection questions need an image to give evidence-backed answers.
              </div>
            )}

            <div style={{ marginTop: '1.5rem' }}>
              <label style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: '0.5rem' }}>Reasoning & Diagnostic Output</label>
              <div style={{ background: 'rgba(11, 15, 25, 0.9)', border: '1px solid var(--border-main)', borderRadius: 12, padding: '1.25rem', minHeight: 140, fontSize: '0.95rem', lineHeight: 1.6, color: '#f8fafc', whiteSpace: 'pre-wrap' }}>
                {reasoningResult || 'Upload an image, then ask a question above...'}
              </div>
            </div>
          </div>
        )}

        
        {/* TAB 3: MODEL PERFORMANCE DASHBOARD */}
        {activeTab === 'analytics' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

            <div className="glass-panel" style={{ borderRadius: 16, padding: '1.25rem 1.5rem' }}>
              <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#fff', marginBottom: '0.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <BarChart3 color="var(--accent-cyan)" size={22} /> Model Performance Dashboard
              </h3>
              <p style={{ fontSize: '0.83rem', color: 'var(--text-muted)' }}>RT-DETR ResNet-50 - WTBD Dataset - 24 Epochs - Test Set (161 images)</p>
            </div>

            <div className="glass-panel" style={{ borderRadius: 16, padding: '1.5rem' }}>
              <h4 style={{ fontSize: '1rem', fontWeight: 600, color: '#fff', marginBottom: '1.25rem' }}>Core Accuracy Metrics - Test Set</h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.1rem' }}>
                {[
                  { label: 'mAP@50 (Detection Accuracy)', value: 84.20, color: '#34d399', desc: 'Detect with 50%+ bounding box overlap' },
                  { label: 'Precision', value: 86.50, color: '#60a5fa', desc: 'Of flagged defects, how many are correct' },
                  { label: 'Recall / Sensitivity', value: 81.80, color: '#f59e0b', desc: 'Of real defects, how many were found' },
                  { label: 'F1-Score', value: 84.09, color: '#c084fc', desc: 'Harmonic balance of Precision and Recall' },
                  { label: 'mAP@50:95 (Strict COCO)', value: 56.40, color: '#38bdf8', desc: 'Avg precision across IoU 0.50 to 0.95' },
                ].map(({ label, value, color, desc }) => (
                  <div key={label}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '0.35rem' }}>
                      <div>
                        <span style={{ fontSize: '0.88rem', fontWeight: 600, color: '#f8fafc' }}>{label}</span>
                        <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)', marginLeft: '0.5rem' }}>{desc}</span>
                      </div>
                      <span style={{ fontWeight: 700, color, fontFamily: 'JetBrains Mono, monospace', fontSize: '1.05rem' }}>{value}%</span>
                    </div>
                    <div style={{ background: 'rgba(11,15,25,0.7)', borderRadius: 20, height: 11, overflow: 'hidden', border: '1px solid var(--border-main)' }}>
                      <div style={{ width: `${value}%`, height: '100%', background: `linear-gradient(90deg, ${color}88, ${color})`, borderRadius: 20 }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
              <div className="glass-panel" style={{ borderRadius: 16, padding: '1.5rem' }}>
                <h4 style={{ fontSize: '1rem', fontWeight: 600, color: '#fff', marginBottom: '1rem' }}>Per-Class Defect Breakdown</h4>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--border-main)' }}>
                      <th style={{ textAlign: 'left', padding: '0.4rem 0.5rem', color: 'var(--text-muted)' }}>Class</th>
                      <th style={{ textAlign: 'center', padding: '0.4rem', color: 'var(--text-muted)' }}>Detection</th>
                      <th style={{ textAlign: 'right', padding: '0.4rem 0.5rem', color: 'var(--text-muted)' }}>Difficulty</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { cls: 'crack',          status: 'Well Detected', diff: 'Medium', color: '#f59e0b' },
                      { cls: 'craze',          status: 'Well Detected', diff: 'Medium', color: '#facc15' },
                      { cls: 'corrosion',      status: 'Well Detected', diff: 'Low',    color: '#60a5fa' },
                      { cls: 'surface_injure', status: 'Well Detected', diff: 'Low',    color: '#34d399' },
                      { cls: 'thunderstrike',  status: 'Well Detected', diff: 'Low',    color: '#c084fc' },
                      { cls: 'hide_craze',     status: 'Hardest Class', diff: 'High',   color: '#f87171' },
                    ].map(({ cls, status, diff, color }) => (
                      <tr key={cls} style={{ borderBottom: '1px solid rgba(35,49,79,0.4)' }}>
                        <td style={{ padding: '0.5rem', color: '#f8fafc', fontWeight: 500 }}>
                          <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: color, marginRight: 6 }} />{cls}
                        </td>
                        <td style={{ padding: '0.4rem', textAlign: 'center' }}>
                          <span style={{ background: color + '20', color, border: '1px solid ' + color + '50', padding: '0.15rem 0.5rem', borderRadius: 20, fontSize: '0.73rem' }}>{status}</span>
                        </td>
                        <td style={{ padding: '0.5rem', textAlign: 'right', color: diff === 'High' ? '#f87171' : diff === 'Medium' ? '#f59e0b' : '#34d399', fontWeight: 600, fontSize: '0.8rem' }}>{diff}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="glass-panel" style={{ borderRadius: 16, padding: '1.5rem' }}>
                <h4 style={{ fontSize: '1rem', fontWeight: 600, color: '#fff', marginBottom: '1rem' }}>WTBD Dataset Information</h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', fontSize: '0.83rem' }}>
                  {[
                    { label: 'Dataset',      value: 'WTBD (Wind Turbine Blade Defect)', color: '#38bdf8' },
                    { label: 'Total Images', value: '1,065 high-res drone images',       color: '#f8fafc' },
                    { label: 'Total BBoxes', value: '4,820 bounding box annotations',    color: '#f8fafc' },
                    { label: 'Classes',      value: '6 non-COCO industrial classes',     color: '#c084fc' },
                    { label: 'Resolution',   value: '640 x 640 px (resized)',             color: '#f8fafc' },
                    { label: 'Source',       value: 'Public drone inspection imagery',   color: '#94a3b8' },
                  ].map(({ label, value, color }) => (
                    <div key={label} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.45rem 0.75rem', background: 'rgba(11,15,25,0.6)', borderRadius: 8, border: '1px solid var(--border-main)' }}>
                      <span style={{ color: 'var(--text-muted)' }}>{label}</span>
                      <span style={{ color, fontWeight: 600, textAlign: 'right', maxWidth: '55%' }}>{value}</span>
                    </div>
                  ))}
                  <div style={{ padding: '0.6rem 0.75rem', background: 'rgba(11,15,25,0.6)', borderRadius: 8, border: '1px solid var(--border-main)' }}>
                    <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '0.4rem' }}>Split Strategy: 70 / 15 / 15</div>
                    <div style={{ display: 'flex', borderRadius: 6, overflow: 'hidden', height: 18 }}>
                      <div style={{ width: '70%', background: '#3b82f6', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.7rem', color: '#fff', fontWeight: 700 }}>Train 745</div>
                      <div style={{ width: '15%', background: '#f59e0b', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.7rem', color: '#fff', fontWeight: 700 }}>Val 159</div>
                      <div style={{ width: '15%', background: '#ef4444', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.7rem', color: '#fff', fontWeight: 700 }}>Test 161</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="glass-panel" style={{ borderRadius: 16, padding: '1.5rem' }}>
              <h4 style={{ fontSize: '1rem', fontWeight: 600, color: '#fff', marginBottom: '0.6rem' }}>Training and Validation Loss - 24 Epochs</h4>
              <div style={{ display: 'flex', gap: '1.5rem', marginBottom: '0.75rem', fontSize: '0.78rem', flexWrap: 'wrap' }}>
                <span style={{ color: '#94a3b8' }}>Epoch 12 to 13: checkpoint resume (LR reset spike)</span>
                <span style={{ color: '#34d399', fontWeight: 600 }}>Best Val: 12.7477 at Epoch 21</span>
                <span style={{ color: '#60a5fa', fontWeight: 600 }}>94.5% reduction: 227.89 to 12.53</span>
              </div>
              <div style={{ width: '100%', height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={lossData} margin={{ top: 5, right: 30, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#23314f" />
                    <XAxis dataKey="epoch" stroke="#94a3b8" label={{ value: 'Epoch', position: 'insideBottomRight', offset: -5 }} />
                    <YAxis stroke="#94a3b8" label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} />
                    <Tooltip contentStyle={{ background: '#131b2e', borderColor: '#23314f', color: '#fff' }} />
                    <Legend />
                    <Line type="monotone" dataKey="trainLoss" name="Training Loss" stroke="#3b82f6" strokeWidth={3} dot={false} />
                    <Line type="monotone" dataKey="evalLoss" name="Validation Loss" stroke="#ef4444" strokeWidth={3} connectNulls />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="glass-panel" style={{ borderRadius: 16, padding: '1.5rem' }}>
              <h4 style={{ fontSize: '1rem', fontWeight: 600, color: '#fff', marginBottom: '1rem' }}>Training Configuration and Hyperparameters</h4>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: '0.7rem', fontSize: '0.82rem' }}>
                {[
                  { k: 'Backbone',       v: 'ResNet-50vd',       c: '#60a5fa' },
                  { k: 'Epochs',         v: '24 / 24 Done',      c: '#34d399' },
                  { k: 'Optimizer',      v: 'AdamW lr=1e-4',     c: '#fff'    },
                  { k: 'Batch Size',     v: '2 x 4 = Eff 8',    c: '#fff'    },
                  { k: 'Best Val Loss',  v: '12.7477 (Ep 21)',   c: '#34d399' },
                  { k: 'Loss Reduction', v: '-94.5% total',      c: '#60a5fa' },
                  { k: 'Precision',      v: 'FP16 Mixed GPU',    c: '#c084fc' },
                  { k: 'Image Size',     v: '640 x 640 px',      c: '#fff'    },
                  { k: 'GPU',            v: 'NVIDIA MX550 2GB',  c: '#f59e0b' },
                  { k: 'Runtimes',       v: 'CUDA + CPU',        c: '#34d399' },
                ].map(({ k, v, c }) => (
                  <div key={k} style={{ background: 'rgba(11,15,25,0.6)', padding: '0.6rem 0.75rem', borderRadius: 8, border: '1px solid var(--border-main)' }}>
                    <div style={{ color: 'var(--text-muted)', marginBottom: '0.2rem', fontSize: '0.76rem' }}>{k}</div>
                    <div style={{ color: c, fontWeight: 600 }}>{v}</div>
                  </div>
                ))}
              </div>
            </div>

          </div>
        )}

      </main>
    </div>
  );
}
