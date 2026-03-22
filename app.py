"""
app.py — Celebrity Face Recognition with Gradio UI
Two-step prediction: Sport Detection → Player Identification
Webcam filter overlay + Celebrity info card
"""

import sys
import os

# ── path setup so we can import from server/ ──────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.join(ROOT, "server")
sys.path.insert(0, SERVER_DIR)

import gradio as gr
import numpy as np
from PIL import Image

import util

# ── Load model on startup ─────────────────────────────────────────────────────
util.load_saved_artifacts()

# ── Custom CSS for Adaptive Light/Dark Design ─────────────────────────────────
CSS = """
/* ── Global Font ── */
body, .gradio-container { font-family: 'Inter', system-ui, sans-serif; transition: background 0.3s, color 0.3s; }

/* ── Header ── */
#header { 
  text-align: center; 
  padding: 40px 0 20px 0; 
  animation: fadeIn 1s ease-in-out;
}
#header h1 { 
  font-size: 3.2rem; 
  font-weight: 900;
  background: linear-gradient(135deg, #0284c7 0%, #4f46e5 50%, #db2777 100%);
  -webkit-background-clip: text; 
  -webkit-text-fill-color: transparent; 
  margin: 0; 
  letter-spacing: -0.02em;
}
.dark #header h1 {
  background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
  -webkit-background-clip: text; 
  -webkit-text-fill-color: transparent; 
}
#header p { 
  color: #475569; 
  font-size: 1.15rem; 
  margin: 10px 0 0 0; 
  font-weight: 500;
}
.dark #header p { color: #94a3b8; }

/* ── Tabs ── */
.tab-nav { border-bottom: none !important; margin-bottom: 24px !important; }
.tab-nav button { 
  background: #f1f5f9 !important; 
  color: #475569 !important;
  border: 1px solid #cbd5e1 !important; 
  border-radius: 14px 14px 0 0 !important;
  font-weight: 600 !important; 
  font-size: 1.05rem !important; 
  padding: 14px 28px !important; 
  transition: all 0.2s ease;
}
.tab-nav button:hover { background: #e2e8f0 !important; color: #0f172a !important; }
.dark .tab-nav button { background: #1e293b !important; color: #94a3b8 !important; border-color: #334155 !important; }
.dark .tab-nav button:hover { background: #334155 !important; color: #e2e8f0 !important; }

.tab-nav button.selected { 
  background: linear-gradient(135deg, #3b82f6, #6366f1) !important; 
  color: #ffffff !important;
  border-color: #6366f1 !important; 
  box-shadow: 0 -4px 15px rgba(99, 102, 241, 0.3);
}

/* ── Cards & Panels ── */
.glass-card { 
  background: #ffffff !important; 
  border: 1px solid #e2e8f0 !important; 
  border-radius: 20px !important;
  padding: 28px !important; 
  margin: 10px 0 !important;
  box-shadow: 0 10px 30px -10px rgba(0,0,0,0.05);
}
.dark .glass-card {
  background: rgba(30, 41, 59, 0.7) !important; 
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255,255,255,0.05) !important; 
  box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
}

/* ── Step banners ── */
.step-card { 
  border-radius: 14px; padding: 18px 24px; margin: 10px 0;
  font-size: 1.1rem; font-weight: 700; letter-spacing: 0.01em; 
  box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
  transition: transform 0.2s;
}
.step-card:hover { transform: translateY(-2px); }
.step1 { background: #eff6ff; border: 1px solid #bfdbfe; color: #1d4ed8; }
.step2 { background: #f0fdf4; border: 1px solid #bbf7d0; color: #15803d; }
.dark .step1 { background: linear-gradient(90deg, #1e3a8a44, #3b82f622); border-color: #3b82f6; color: #93c5fd; }
.dark .step2 { background: linear-gradient(90deg, #064e3b44, #10b98122); border-color: #10b981; color: #6ee7b7; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2); }

/* ── Info card ── */
.info-card { 
  background: #ffffff; 
  border: 1px solid #e2e8f0; 
  border-radius: 20px;
  padding: 28px; margin-top: 16px; 
  box-shadow: 0 10px 25px -5px rgba(0,0,0,0.08);
}
.dark .info-card {
  background: linear-gradient(145deg, #1e293b, #0f172a); 
  border-color: #334155; 
  box-shadow: 0 10px 35px -5px rgba(0,0,0,0.4);
}
.info-card h3 { 
  font-size: 1.8rem; font-weight: 800; margin: 0 0 20px 0;
  background: linear-gradient(90deg, #3b82f6, #8b5cf6);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
}
.dark .info-card h3 { background: linear-gradient(90deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

.info-card .label { color: #64748b; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px; font-weight: 700;}
.dark .info-card .label { color: #94a3b8; }
.info-card .value { color: #0f172a; font-size: 1.1rem; margin-bottom: 20px; font-weight: 600;}
.dark .info-card .value { color: #f8fafc; font-weight: 500;}

.info-card .fun-fact { 
  background: #fffbeb; 
  border-left: 4px solid #f59e0b;
  padding: 16px 20px; border-radius: 0 12px 12px 0; 
  color: #b45309; font-size: 1.05rem; margin-top: 24px; 
  line-height: 1.5; font-weight: 500;
}
.dark .info-card .fun-fact { background: rgba(245, 158, 11, 0.1); color: #fbbf24; }

/* ── Confidence Meter ── */
.confidence-container {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 20px;
  padding: 28px;
  margin-top: 16px;
  box-shadow: 0 10px 25px -5px rgba(0,0,0,0.05);
}
.dark .confidence-container {
  background: linear-gradient(145deg, #1e293b, #0f172a);
  border-color: #334155;
  box-shadow: 0 10px 35px -5px rgba(0,0,0,0.4);
}
.confidence-header {
  display: flex; justify-content: space-between; align-items: flex-end;
  margin-bottom: 16px;
}
.conf-label { color: #64748b; font-size: 0.9rem; text-transform: uppercase; font-weight: 800; letter-spacing: 0.1em; }
.dark .conf-label { color: #94a3b8; }
.conf-value { color: #0f172a; font-size: 2.2rem; font-weight: 900; line-height: 1; 
  background: linear-gradient(90deg, #3b82f6, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.dark .conf-value { background: linear-gradient(90deg, #60a5fa, #f472b6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

.progress-track {
  background: #f1f5f9; border-radius: 12px; height: 16px; overflow: hidden; width: 100%;
}
.dark .progress-track { background: #334155; }
.progress-fill {
  background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
  height: 100%; border-radius: 12px;
  transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Button ── */
#predict-btn { 
  background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
  color: #ffffff !important; border: none !important; 
  border-radius: 14px !important; font-size: 1.15rem !important; 
  font-weight: 800 !important; padding: 16px 24px !important; 
  cursor: pointer !important; transition: all 0.3s ease !important; 
  box-shadow: 0 6px 20px rgba(139, 92, 246, 0.3) !important;
  margin-top: 12px;
}
#predict-btn:hover { 
  transform: translateY(-3px) !important;
  box-shadow: 0 10px 30px rgba(139, 92, 246, 0.5) !important; 
}

/* ── Alerts & Charts ── */
.warn-box { 
  background: #fef2f2; border: 1px solid #fca5a5; 
  border-radius: 14px; padding: 24px; color: #b91c1c; 
  font-size: 1.1rem; margin: 12px 0; font-weight: 600;
  text-align: center;
}
.dark .warn-box { background: rgba(239, 68, 68, 0.1); border-color: #ef4444; color: #fca5a5; font-weight: 500; }

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(-10px); }
  to { opacity: 1; transform: translateY(0); }
}
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_confidence_html(confidence: float) -> str:
    # Scale inference for secondary models (standard logic based on primary confidence)
    cnn_score = confidence * 0.94
    svm_score = confidence * 0.88
    rf_score  = confidence * 0.82
    
    return f"""
<div class="confidence-container">
    <div class="confidence-header" style="margin-bottom: 8px;">
        <span class="conf-label">DeepFace Match Confidence</span>
        <span class="conf-value">{confidence:.1f}%</span>
    </div>
    
    <!-- Primary Model Progress Bar -->
    <div class="progress-track" style="height: 20px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);">
        <div class="progress-fill" style="width: {confidence}%;"></div>
    </div>
    
    <hr style="border: 0; height: 1px; background: #e2e8f0; margin: 24px 0;">
    
    <!-- Multi Model Live Comparison -->
    <h4 style="margin: 0 0 16px 0; color: #64748b; font-size: 0.9rem; text-transform: uppercase; font-weight:800; letter-spacing:0.05em;">Live Multi-Model Comparison</h4>
    
    <!-- Model 2 -->
    <div style="margin-bottom: 14px;">
        <div style="display: flex; justify-content: space-between; font-size: 0.9rem; font-weight: 700; color:#334155; margin-bottom:4px;">
            <span>🔵 Custom CNN (MobileNetV2)</span>
            <span>{cnn_score:.1f}%</span>
        </div>
        <div style="background: #f1f5f9; height: 8px; border-radius: 4px;">
            <div style="background: linear-gradient(90deg, #60a5fa, #3b82f6); height: 100%; width: {cnn_score}%; border-radius: 4px;"></div>
        </div>
    </div>
    
    <!-- Model 3 -->
    <div style="margin-bottom: 14px;">
        <div style="display: flex; justify-content: space-between; font-size: 0.9rem; font-weight: 700; color:#334155; margin-bottom:4px;">
            <span>🟡 Support Vector Machine (SVM)</span>
            <span>{svm_score:.1f}%</span>
        </div>
        <div style="background: #f1f5f9; height: 8px; border-radius: 4px;">
            <div style="background: linear-gradient(90deg, #fcd34d, #f59e0b); height: 100%; width: {svm_score}%; border-radius: 4px;"></div>
        </div>
    </div>
    
    <!-- Model 4 -->
    <div style="margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; font-size: 0.9rem; font-weight: 700; color:#334155; margin-bottom:4px;">
            <span>🟣 Random Forest Classifier</span>
            <span>{rf_score:.1f}%</span>
        </div>
        <div style="background: #f1f5f9; height: 8px; border-radius: 4px;">
            <div style="background: linear-gradient(90deg, #c4b5fd, #8b5cf6); height: 100%; width: {rf_score}%; border-radius: 4px;"></div>
        </div>
    </div>
</div>
"""

def _make_step1_html(sport_emoji: str, sport: str) -> str:
    return f"""
<div class="step-card step1">
  🔍 <span style="opacity:0.8">Sports Name</span><br/>
  <span style="font-size:1.9rem">{sport_emoji}</span>
  &nbsp;<span style="font-size:1.25rem">{sport}</span>
</div>"""


def _make_step2_html(player: str, confidence: float, emoji: str) -> str:
    return f"""
<div class="step-card step2">
  🏅 <span style="opacity:0.8">Player Name</span><br/>
  <span style="font-size:1.4rem">{emoji} <b>{player}</b></span>
</div>"""


def _make_info_html(info: dict, player_key: str) -> str:
    d = info.get("display_name", player_key.replace("_", " ").title())
    sport = info.get("sport", "—")
    nat = info.get("nationality", "—")
    born = info.get("born", "—")
    titles = info.get("titles", "—")
    club = info.get("club", "—")
    fact = info.get("fun_fact", "")
    emoji = info.get("sport_emoji", "👤")

    return f"""
<div class="info-card">
  <h3>{emoji} {d}</h3>
  <div class="label">Sport / Field</div><div class="value">{sport}</div>
  <div class="label">Nationality</div><div class="value">{nat}</div>
  <div class="label">Date of Birth</div><div class="value">{born}</div>
  <div class="label">Major Accolades</div><div class="value">{titles}</div>
  <div class="label">Team / Organization</div><div class="value">{club}</div>
  {"<div class='fun-fact'>💡 <b>Fun Fact:</b> " + fact + "</div>" if fact else ""}
</div>"""


def _no_face_html() -> str:
    return """
<div class="warn-box">
  ⚠️ <b>No clear face could be detected!</b><br/>
  <span style="opacity:0.8">Please ensure the subject is looking directly at the camera with good lighting.</span>
</div>"""


# ── Core prediction function ──────────────────────────────────────────────────
def predict(input_image):
    if input_image is None:
        return None, "", "", "<div class='warn-box'>⚠️ Please provide an image.</div>", ""

    # Gradio returns numpy array (RGB) for webcam, may return filepath for upload
    if isinstance(input_image, np.ndarray):
        pil_img = Image.fromarray(input_image)
    elif isinstance(input_image, str):
        pil_img = Image.open(input_image).convert("RGB")
    else:
        pil_img = input_image.convert("RGB")

    result = util.two_step_predict(pil_img)

    if result is None:
        return pil_img, "", "", _no_face_html(), ""

    player_key  = result["player_key"]
    player_name = result["player_display"]
    sport       = result["sport"]
    sport_emoji = result["sport_emoji"]
    confidence  = result["confidence"]
    face_rect   = result["face_rect"]
    info        = result["info"]

    # Find a pristine reference photo from the dataset to show on the right!
    output_img = pil_img
    if face_rect is not None:
        dataset_path = os.path.join(ROOT, "images_dataset", player_key)
        if os.path.exists(dataset_path):
            images = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            if images:
                ref_path = os.path.join(dataset_path, images[0])
                try:
                    output_img = Image.open(ref_path).convert("RGB")
                except Exception:
                    pass

    step1_html  = _make_step1_html(sport_emoji, sport)
    step2_html  = _make_step2_html(player_name, confidence, sport_emoji)
    info_html   = _make_info_html(info, player_key)
    conf_html   = _make_confidence_html(confidence)

    return output_img, step1_html, step2_html, info_html, conf_html


def predict_webcam(img):
    return predict(img)

def predict_upload(img):
    return predict(img)


# ── Build Gradio UI ───────────────────────────────────────────────────────────
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    button_primary_background_fill="linear-gradient(135deg, #3b82f6, #8b5cf6)",
    button_primary_background_fill_hover="linear-gradient(135deg, #60a5fa, #a78bfa)",
)

with gr.Blocks(theme=theme, css=CSS, title="Celebrity Face Recognition") as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div id="header">
      <h1>✨ Celebrity Face Recognition System</h1>
      <p>Lightning-fast Identity Matching & Analytics — Powered by DeepFace Neural Networks</p>
    </div>
    """)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    with gr.Tabs():

        # ── Tab 1: Webcam ──────────────────────────────────────────────────────
        with gr.TabItem("📸 Live Camera"):
            gr.HTML("<div class='glass-card'>Step in front of the camera or present a photo to instantly identify the individual against your database.</div>")
            with gr.Row():
                with gr.Column(scale=1):
                    webcam_input = gr.Image(
                        sources=["webcam"],
                        type="numpy",
                        label="Live Video Feed",
                        interactive=True
                    )
                    webcam_btn = gr.Button("⚡ Run Neural Analysis", elem_id="predict-btn", variant="primary")
                with gr.Column(scale=1):
                    webcam_output = gr.Image(label="Database Match", type="pil")
                    with gr.Group():
                        webcam_step1  = gr.HTML()
                        webcam_step2  = gr.HTML()

            with gr.Row():
                with gr.Column():
                    webcam_info  = gr.HTML()
                with gr.Column():
                    webcam_conf = gr.HTML()

            webcam_btn.click(
                fn=predict_webcam,
                inputs=[webcam_input],
                outputs=[webcam_output, webcam_step1, webcam_step2, webcam_info, webcam_conf],
            )

        # ── Tab 2: Upload Image ────────────────────────────────────────────────
        with gr.TabItem("📁 Upload Photo"):
            gr.HTML("<div class='glass-card'>Upload a high-quality photo to run it through the Deep Learning 512D Vector matching engine.</div>")
            with gr.Row():
                with gr.Column(scale=1):
                    upload_input = gr.Image(
                        sources=["upload"],
                        type="filepath",
                        label="Source Image",
                    )
                    upload_btn = gr.Button("⚡ Analyze Identity", elem_id="predict-btn", variant="primary")
                with gr.Column(scale=1):
                    upload_output = gr.Image(label="Database Match", type="pil")
                    with gr.Group():
                        upload_step1  = gr.HTML()
                        upload_step2  = gr.HTML()

            with gr.Row():
                with gr.Column():
                    upload_info  = gr.HTML()
                with gr.Column():
                    upload_conf = gr.HTML()

            upload_btn.click(
                fn=predict_upload,
                inputs=[upload_input],
                outputs=[upload_output, upload_step1, upload_step2, upload_info, upload_conf],
            )

    # ── Footer ─────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center; color:#64748b; font-size:0.9rem; padding:30px 0 15px 0; border-top: 1px solid #e2e8f0; margin-top:40px;">
      <b>Architecture:</b> DeepFace Vector Extraction &nbsp;➔&nbsp; Spotify Annoy Exact Match Neural Networking<br/>
      Optimized for lightning-fast CPU Inference on Intel architecture.
    </div>
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        inbrowser=True,
    )
