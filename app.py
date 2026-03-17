import streamlit as st
import cv2
from ultralytics import YOLO
import time
import numpy as np
import plotly.graph_objects as go
import os
import pandas as pd
from plotly.subplots import make_subplots
from pathlib import Path
import math

"""
Pineapple Inference Dashboard — Folder Mode
============================================
Loads images from a local folder, runs YOLO inference on each,
computes distances / pass-fail, and displays everything in the
same dashboard layout as the DB version — no database needed.
"""

# ─────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE_FOLDER = os.path.join(BASE_DIR, "assets", "foto")
DEFAULT_MODEL_PATH   = os.path.join(BASE_DIR, "assets", "train7", "weights", "best.pt")

CONF_THRESHOLD  = 0.5
POLL_INTERVAL   = 3.75   # seconds between images in loop mode
MAX_RATE        = 16     # max units / minute (for throughput graph)

# Real-world plateau dimensions for mm calibration
PLATEAU_REAL_W_MM = 120.0
PLATEAU_REAL_H_MM = 120.0

# Class name → index mapping (must match your model)
IDX_ANANAS   = "ananas"
IDX_KERN     = "core"
IDX_GEVULD   = "gevuld"
IDX_ONGEVULD = "ongevuld"
IDX_PLATEAU  = "plateau"

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ─────────────────────────────────────────
# PAGE & STYLE
# ─────────────────────────────────────────
st.set_page_config(page_title="Pineapple Monitor", page_icon="🍍", layout="wide")


# ─────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────
@st.cache_resource
def load_model(path: str):
    if not Path(path).exists():
        st.error(f"Model not found: {path}")
        st.stop()
    model = YOLO(path)
    model.to("cpu")
    return model

# ─────────────────────────────────────────
# IMAGE FOLDER HELPERS
# ─────────────────────────────────────────
def get_image_files(folder: str) -> list[Path]:
    p = Path(folder)
    if not p.exists():
        return []
    return sorted([f for f in p.iterdir() if f.suffix.lower() in SUPPORTED_EXTS])

def load_image(path: Path) -> np.ndarray | None:
    """Load image file → BGR numpy array."""
    img = cv2.imread(str(path))
    return img  # BGR

# ─────────────────────────────────────────
# INFERENCE & CALCULATIONS
# ─────────────────────────────────────────
def run_inference(model, img_bgr: np.ndarray, conf: float) -> dict:
    """
    Run YOLO on a BGR image.
    Returns dict of {class_name: (cx, cy, w, h, conf)} for the
    highest-confidence detection of each relevant class.
    """
    results  = model(img_bgr, conf=conf, verbose=False)
    boxes    = results[0].boxes
    names    = model.names
    best     = {}

    if boxes is None or len(boxes) == 0:
        return best

    for box in boxes:
        cls_idx   = int(box.cls[0])
        cls_name  = names[cls_idx].lower()
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w  = x2 - x1
        h  = y2 - y1

        if cls_name not in best or confidence > best[cls_name][4]:
            best[cls_name] = (cx, cy, w, h, confidence)

    return best


def compute_distances_px(detections: dict) -> dict:
    """
    Compute edge-to-edge distances between detected objects (in pixels).
    Returns a flat dict matching the DB column naming convention.
    """
    dist = {}

    def edges(name):
        d = detections.get(name)
        if d is None:
            return None
        cx, cy, w, h, _ = d
        return dict(left=cx-w/2, right=cx+w/2, top=cy-h/2, bottom=cy+h/2, cx=cx, cy=cy)

    ananas  = edges(IDX_ANANAS)
    kern    = edges(IDX_KERN)
    plateau = edges(IDX_PLATEAU)

    def gap(a_edge, b_edge):
        return abs(a_edge - b_edge)

    if ananas and plateau:
        dist["dist_ananas_plateau_left_px"]   = gap(ananas["left"],   plateau["left"])
        dist["dist_ananas_plateau_right_px"]  = gap(ananas["right"],  plateau["right"])
        dist["dist_ananas_plateau_top_px"]    = gap(ananas["top"],    plateau["top"])
        dist["dist_ananas_plateau_bottom_px"] = gap(ananas["bottom"], plateau["bottom"])

    if kern and plateau:
        dist["dist_kern_plateau_left_px"]   = gap(kern["left"],   plateau["left"])
        dist["dist_kern_plateau_right_px"]  = gap(kern["right"],  plateau["right"])
        dist["dist_kern_plateau_top_px"]    = gap(kern["top"],    plateau["top"])
        dist["dist_kern_plateau_bottom_px"] = gap(kern["bottom"], plateau["bottom"])

    if kern and ananas:
        dist["dist_kern_ananas_left_px"]   = kern["cx"] - ananas["left"]
        dist["dist_kern_ananas_right_px"]  = ananas["right"] - kern["cx"]
        dist["dist_kern_ananas_top_px"]    = kern["cy"] - ananas["top"]
        dist["dist_kern_ananas_bottom_px"] = ananas["bottom"] - kern["cy"]

    if ananas:
        dist["opp"] = math.pi * ((ananas["bottom"] - ananas["top"]) / 2) * ((ananas["right"] - ananas["left"]) / 2)

    return dist


def apply_mm_scale(distances_px: dict, scale_x: float, scale_y: float) -> dict:
    """Convert px distances to mm using calibration scale."""
    mm = {}
    for key, val in distances_px.items():
        mm_key = key.replace("_px", "_mm")
        # Use scale_x for horizontal, scale_y for vertical distances
        if "left" in key or "right" in key:
            mm[mm_key] = val * scale_x
        else:
            mm[mm_key] = val * scale_y
    return mm


def evaluate_pass_fail(detections: dict, distances_px: dict,
                       scale_x: float | None, scale_y: float | None) -> tuple:
    """
    Simple pass/fail logic — adapt thresholds to your requirements.
    Returns (pass_fail: int, fail_reason: str | None)
    """
    # Must have ananas detected
    # Plateau must always be detected for any measurement to make sense
    if IDX_PLATEAU not in detections:
        return 0, "Plateau niet gedetecteerd"

    # Ongevuld plateau: no ananas/kern/gevuld expected — this is a known valid empty state
    if IDX_ONGEVULD in detections:
        return 0, "Plateau is leeg (Ongevuld)"

    # From here on we expect an ananas
    if IDX_ANANAS not in detections:
        return 0, "Ananas niet gedetecteerd"

    # Gevuld but no ananas fill detected
    if IDX_GEVULD not in detections:
        return 0, "Gevuld niet gedetecteerd"

    # If kern detected, check centering
    if IDX_KERN in detections and scale_x and scale_y:
        kern_ananas_dists = [
            distances_px.get("dist_kern_ananas_left_px", 0)   * scale_x,
            distances_px.get("dist_kern_ananas_right_px", 0)  * scale_x,
            distances_px.get("dist_kern_ananas_top_px", 0)    * scale_y,
            distances_px.get("dist_kern_ananas_bottom_px", 0) * scale_y,
        ]
        d_min = min(kern_ananas_dists)
        if d_min < 5.0:   # < 5 mm margin → fail
            return 0, f"Kern te dicht bij rand ({d_min:.1f} mm)"

    return 1, None


def build_row(img_path: Path, img_bgr: np.ndarray,
              detections: dict, distances_px: dict) -> dict:
    """Build a flat dict identical in structure to what the DB version returns."""
    plateau = detections.get(IDX_PLATEAU)
    scale_x = (PLATEAU_REAL_W_MM / plateau[2]) if plateau and plateau[2] else None
    scale_y = (PLATEAU_REAL_H_MM / plateau[3]) if plateau and plateau[3] else None

    distances_mm = apply_mm_scale(distances_px, scale_x or 0, scale_y or 0) \
                   if scale_x and scale_y else {}

    pass_fail, fail_reason = evaluate_pass_fail(detections, distances_px, scale_x, scale_y)

    def det(name, field):
        d = detections.get(name)
        if d is None: return None
        return d[{"cx":0,"cy":1,"w":2,"h":3,"conf":4}[field]]

    return {
        "id":            img_path.name,
        "triggered_at":  pd.Timestamp.now(),
        "image_path":    str(img_path),

        "ananas_detected":   int(IDX_ANANAS   in detections),
        "kern_detected":     int(IDX_KERN     in detections),
        "gevuld_detected":   int(IDX_GEVULD   in detections),
        "ongevuld_detected": int(IDX_ONGEVULD in detections),
        "plateau_detected":  int(IDX_PLATEAU  in detections),

        "ananas_conf":   det(IDX_ANANAS,  "conf"),
        "kern_conf":     det(IDX_KERN,    "conf"),
        "gevuld_conf":   det(IDX_GEVULD,  "conf"),
        "ongevuld_conf": det(IDX_ONGEVULD,"conf"),
        "plateau_conf":  det(IDX_PLATEAU, "conf"),

        "ananas_cx_px":  det(IDX_ANANAS,  "cx"),
        "ananas_cy_px":  det(IDX_ANANAS,  "cy"),
        "ananas_w_px":   det(IDX_ANANAS,  "w"),
        "ananas_h_px":   det(IDX_ANANAS,  "h"),
        "kern_cx_px":    det(IDX_KERN,    "cx"),
        "kern_cy_px":    det(IDX_KERN,    "cy"),
        "kern_w_px":     det(IDX_KERN,    "w"),
        "kern_h_px":     det(IDX_KERN,    "h"),
        "plateau_cx_px": det(IDX_PLATEAU, "cx"),
        "plateau_cy_px": det(IDX_PLATEAU, "cy"),
        "plateau_w_px":  det(IDX_PLATEAU, "w"),
        "plateau_h_px":  det(IDX_PLATEAU, "h"),

        "scale_x_mm_per_px": scale_x,
        "scale_y_mm_per_px": scale_y,

        **distances_px,
        **distances_mm,

        "pass_fail":   pass_fail,
        "fail_reason": fail_reason,

        # Keep raw BGR for display
        "_img_bgr": img_bgr,
    }

# ─────────────────────────────────────────
# DRAWING
# ─────────────────────────────────────────
BLANK = np.zeros((200, 200, 3), dtype=np.uint8)

def draw_detections(img_bgr: np.ndarray, row: dict) -> np.ndarray:
    img = img_bgr.copy()
    objects = [
        (IDX_ANANAS,  "ananas_cx_px",  "ananas_cy_px",  "ananas_w_px",  "ananas_h_px",  "ananas_conf",  (0, 200, 255)),
        (IDX_KERN,    "kern_cx_px",    "kern_cy_px",    "kern_w_px",    "kern_h_px",    "kern_conf",    (255, 180, 0)),
        (IDX_PLATEAU, "plateau_cx_px", "plateau_cy_px", "plateau_w_px", "plateau_h_px", None,           (180, 180, 180)),
    ]
    for name, cx_k, cy_k, w_k, h_k, conf_k, color in objects:
        cx, cy, w, h = row.get(cx_k), row.get(cy_k), row.get(w_k), row.get(h_k)
        if None in (cx, cy, w, h): continue
        x1, y1 = int(cx - w/2), int(cy - h/2)
        x2, y2 = int(cx + w/2), int(cy + h/2)
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        conf = row.get(conf_k)
        label = f"{name} {conf:.2f}" if conf else name
        cv2.putText(img, label, (x1, max(y1-6,10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    pf    = row.get("pass_fail")
    stamp = "PASS" if pf == 1 else "FAIL" if pf == 0 else "?"
    col   = (0, 230, 0) if pf == 1 else (0, 0, 230)
    cv2.putText(img, stamp, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
    if row.get("fail_reason"):
        cv2.putText(img, str(row["fail_reason"]), (8, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 230), 1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ─────────────────────────────────────────
# DERIVED COLUMNS
# ─────────────────────────────────────────
def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df = df.copy()
    sx = df["scale_x_mm_per_px"] if "scale_x_mm_per_px" in df.columns else pd.Series(np.nan, index=df.index)
    sy = df["scale_y_mm_per_px"] if "scale_y_mm_per_px" in df.columns else pd.Series(np.nan, index=df.index)

    # delta_x/y only meaningful when both kern and ananas are present
    kern_cx   = df["kern_cx_px"]   if "kern_cx_px"   in df.columns else pd.Series(np.nan, index=df.index)
    kern_cy   = df["kern_cy_px"]   if "kern_cy_px"   in df.columns else pd.Series(np.nan, index=df.index)
    ananas_cx = df["ananas_cx_px"] if "ananas_cx_px" in df.columns else pd.Series(np.nan, index=df.index)
    ananas_cy = df["ananas_cy_px"] if "ananas_cy_px" in df.columns else pd.Series(np.nan, index=df.index)

    df["delta_x_mm"] = (kern_cx - ananas_cx) * sx
    df["delta_y_mm"] = (kern_cy - ananas_cy) * sy

    # d_min only meaningful when kern detected
    dist_cols = ["dist_kern_ananas_left_mm","dist_kern_ananas_right_mm",
                 "dist_kern_ananas_top_mm", "dist_kern_ananas_bottom_mm"]
    existing = [c for c in dist_cols if c in df.columns]
    if existing:
        df["d_min_mm"] = df[existing].min(axis=1)
    else:
        df["d_min_mm"] = np.nan
    return df

# ─────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────
BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
    font=dict(family="Share Tech Mono", color="#19191a", size=11),
    title_font=dict(family="Share Tech Mono", color="#00e5ff", size=13),
    xaxis=dict(gridcolor="#ffffff", linecolor="#000000"),
    yaxis=dict(gridcolor="#ffffff", linecolor="#000000"),
    margin=dict(l=40,r=20,t=40,b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2a3a"),
)
CLASS_COLORS = {"Goed":"#39ff14","Risicozone":"#ffa502","Fout":"#ff4757"}

def classify(df, drill_r, margin):
    if "d_min_mm" not in df.columns:
        df = df.copy()
        df["d_min_mm"] = np.nan
        df["cls"] = "?"
        return df
    d = df.dropna(subset=["d_min_mm"]).copy()
    if d.empty:
        d["cls"] = "?"
        return d
    d["cls"] = np.select(
        [d["d_min_mm"] >= drill_r+margin,
         (d["d_min_mm"] >= drill_r) & (d["d_min_mm"] < drill_r+margin),
         d["d_min_mm"] < drill_r],
        ["Goed","Risicozone","Fout"], default="?")
    return d

# ─────────────────────────────────────────
# GRAPHS
# ─────────────────────────────────────────
def graph_throughput(df):
    fig = go.Figure()
    if df.empty or "triggered_at" not in df:
        return fig.update_layout(**BASE)

    # Only count rows where ananas was detected (exclude Ongevuld)
    ananas_df = df[df.get("ananas_detected", pd.Series(0, index=df.index)) == 1].copy()

    if ananas_df.empty:
        return fig.update_layout(**{**BASE, "title":"Throughput — Ananas per Minute"})

    per_min = ananas_df.set_index("triggered_at").resample("1min").size().reset_index(name="count")
    per_min["max"]        = MAX_RATE
    per_min["cum_actual"] = per_min["count"].cumsum()
    per_min["cum_max"]    = per_min["max"].cumsum()
    per_min["pct"]        = (per_min["cum_actual"] / per_min["cum_max"] * 100).round(1)
    fig.add_trace(go.Bar(x=per_min["triggered_at"], y=per_min["count"],
                         name="Ananas", marker_color="#00e5ff", opacity=0.8,
                         customdata=np.stack([per_min["cum_actual"],per_min["pct"]],axis=-1),
                         hovertemplate="Time: %{x}<br>Ananas: %{y}<br>Cumulative: %{customdata[0]}<br>Fill: %{customdata[1]}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=per_min["triggered_at"], y=per_min["max"],
                             name="Max", mode="lines",
                             line=dict(color="#ff4757", dash="dash", width=2)))
    fig.update_layout(**{**BASE,"title":"Throughput — Ananas per Minute (Ongevuld uitgesloten)",
                         "xaxis_title":"Time","yaxis_title":"Ananas / min",
                         "height":280,"hovermode":"x unified"})
    return fig

def graph_centering_scatter(df, drill_r, margin):
    fig = go.Figure()
    if df.empty or "delta_x_mm" not in df:
        return fig.update_layout(**BASE)
    d = classify(df.dropna(subset=["delta_x_mm","delta_y_mm"]), drill_r, margin)
    for label, grp in d.groupby("cls"):
        fig.add_trace(go.Scatter(x=grp["delta_x_mm"], y=grp["delta_y_mm"],
                                 mode="markers", name=label,
                                 marker=dict(size=9,color=CLASS_COLORS.get(label,"#888"),opacity=0.8),
                                 hovertemplate="Δx: %{x:.1f} mm<br>Δy: %{y:.1f} mm<extra></extra>"))
    fig.add_hline(y=0, line_dash="dash", line_color="#334455")
    fig.add_vline(x=0, line_dash="dash", line_color="#334455")
    fig.update_layout(**{**BASE,"title":"Kern Centering — Δx vs Δy (mm)",
                         "xaxis_title":"Δx (mm)","yaxis_title":"Δy (mm)",
                         "yaxis_scaleanchor":"x","height":350})
    return fig

def graph_dmin(df, drill_r, margin):
    fig = go.Figure()
    if df.empty or "d_min_mm" not in df:
        return fig.update_layout(**BASE)
    d = classify(df, drill_r, margin)
    fig.add_trace(go.Scatter(x=d.index, y=d["d_min_mm"], mode="lines",
                             line=dict(color="#334455",width=1), showlegend=False))
    for label, grp in d.groupby("cls"):
        fig.add_trace(go.Scatter(x=grp.index, y=grp["d_min_mm"], mode="markers",
                                 name=label, marker=dict(size=8,color=CLASS_COLORS.get(label,"#888")),
                                 hovertemplate="Product #%{x}<br>D_min: %{y:.1f} mm<extra></extra>"))
    fig.add_hline(y=drill_r,         line_dash="dash", line_color="#ff4757",
                  annotation_text="Boorstraal", annotation_font_color="#ff4757")
    fig.add_hline(y=drill_r+margin,  line_dash="dash", line_color="#ffa502",
                  annotation_text="Marge",       annotation_font_color="#ffa502")
    fig.update_layout(**{**BASE,"title":"D_min per Product (mm)",
                         "xaxis_title":"Product #","yaxis_title":"D_min (mm)",
                         "yaxis_autorange":"reversed","height":300})
    return fig

def graph_delta_ts(df):
    if df.empty or "delta_x_mm" not in df:
        return go.Figure().update_layout(**BASE)
    d  = df.dropna(subset=["delta_x_mm","delta_y_mm"]).copy()
    sx, sy = d["delta_x_mm"].std(), d["delta_y_mm"].std()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["Δx kern vs ananas (mm)","Δy kern vs ananas (mm)"],
                        vertical_spacing=0.14)
    fig.add_trace(go.Scatter(x=d.index, y=d["delta_x_mm"], mode="lines+markers",
                             line=dict(color="#00e5ff",width=1.5), name="Δx"), row=1, col=1)
    for s in [2*sx, -2*sx]:
        fig.add_hline(y=s, line_dash="dash", line_color="#334455", row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["delta_y_mm"], mode="lines+markers",
                             line=dict(color="#ffa502",width=1.5), name="Δy"), row=2, col=1)
    for s in [2*sy, -2*sy]:
        fig.add_hline(y=s, line_dash="dash", line_color="#334455", row=2, col=1)
    fig.update_layout(**{**BASE,"title":"Kern Offset over Tijd (±2σ)","height":420,"showlegend":True})
    fig.update_xaxes(gridcolor="#ffffff", linecolor="#000000")
    fig.update_yaxes(gridcolor="#ffffff", linecolor="#000000")
    return fig

def graph_confidence(df):
    fig = go.Figure()
    if df.empty: return fig.update_layout(**BASE)
    for col, color, label in [("ananas_conf","#00e5ff","Ananas"),
                               ("kern_conf",  "#ffa502","Kern"),
                               ("gevuld_conf","#39ff14","Gevuld")]:
        if col in df.columns:
            d = df.dropna(subset=[col])
            fig.add_trace(go.Scatter(x=d.index, y=d[col], mode="lines", name=label,
                                     line=dict(color=color, width=1.5),
                                     hovertemplate=f"{label}: %{{y:.3f}}<extra></extra>"))
    fig.update_layout(**{**BASE,"title":"Detection Confidence over Tijd",
                         "xaxis_title":"Product #","yaxis_title":"Confidence",
                         "yaxis_range":[0,1],"height":260})
    return fig

def graph_distances(df):
    fig = go.Figure()
    if df.empty: return fig.update_layout(**BASE)
    for col, color, label in [
        ("dist_ananas_plateau_left_mm",  "#00e5ff","Left"),
        ("dist_ananas_plateau_right_mm", "#ffa502","Right"),
        ("dist_ananas_plateau_top_mm",   "#39ff14","Top"),
        ("dist_ananas_plateau_bottom_mm","#ff4757","Bottom"),
    ]:
        if col in df.columns:
            d = df.dropna(subset=[col])
            fig.add_trace(go.Scatter(x=d.index, y=d[col], mode="lines", name=label,
                                     line=dict(color=color, width=1.5)))
    fig.update_layout(**{**BASE,"title":"Ananas → Plateau Distances (mm)",
                         "xaxis_title":"Product #","yaxis_title":"Distance (mm)","height":280})
    return fig

def graph_opp(df):
    fig = go.Figure()
    if df.empty or "opp" not in df.columns:
        return fig.update_layout(**BASE)
    d = df.dropna(subset=["opp"])
    if d.empty:
        return fig.update_layout(**BASE)
    fig.add_trace(go.Scatter(
        x=d.index, y=d["opp"],
        mode="lines+markers",
        line=dict(color="#00e5ff", width=1.5),
        marker=dict(size=6),
        hovertemplate="Product #%{x}<br>Opp: %{y:.1f} px²<extra></extra>"
    ))
    fig.update_layout(**{**BASE,
        "title": "Ananas Oppervlakte (px²)",
        "xaxis_title": "Product #",
        "yaxis_title": "Oppervlakte (px²)",
        "height": 260
    })
    return fig


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙ SETTINGS")
    st.markdown("---")

    image_folder = st.text_input("Image folder", value=DEFAULT_IMAGE_FOLDER)
    model_path   = st.text_input("Model path",   value=DEFAULT_MODEL_PATH)
    conf_thresh  = st.slider("Confidence threshold", 0.1, 1.0, CONF_THRESHOLD, step=0.05)
    loop_speed   = st.slider("Interval (s)", 0.5, 10.0, POLL_INTERVAL, step=0.25)

    st.markdown("---")
    st.markdown("### Centering Thresholds")
    drill_r = st.selectbox("Drill radius (mm)", [25, 30, 35, 40], index=2)
    margin  = st.selectbox("Margin (mm)",       [3, 5, 7, 10],   index=1)

    st.markdown("---")
    if st.button("↺ Reset loop"):
        st.session_state.folder_index = 0
        st.session_state.folder_done  = False
        st.session_state.history      = []
        st.rerun()

    st.markdown("---")
    st.markdown('<div style="font-family:Share Tech Mono,monospace;font-size:10px;'
                'color:#334455;text-align:center;">PINEAPPLE MONITOR — FOLDER MODE</div>',
                unsafe_allow_html=True)

# ─────────────────────────────────────────
# STATIC LAYOUT
# ─────────────────────────────────────────
st.markdown("# 🍍 PINEAPPLE INFERENCE MONITOR")
status_ph = st.empty()

ic1, ic2 = st.columns(2)
with ic1:
    st.markdown('<div class="img-label">RAW IMAGE</div>', unsafe_allow_html=True)
    raw_ph = st.empty()
with ic2:
    st.markdown('<div class="img-label">INFERENCE — BOUNDING BOXES</div>', unsafe_allow_html=True)
    inf_ph = st.empty()

info_ph = st.empty()
st.markdown("---")

kpi = [c.empty() for c in st.columns(6)]
st.markdown("---")

st.markdown('<div class="section-title">THROUGHPUT</div>', unsafe_allow_html=True)
throughput_ph = st.empty()

st.markdown('<div class="section-title">CENTERING ANALYSIS</div>', unsafe_allow_html=True)
ga, gb = st.columns(2)
with ga: scatter_ph = st.empty()
with gb: dmin_ph    = st.empty()

st.markdown('<div class="section-title">POSITIE OFFSET OVER TIJD</div>', unsafe_allow_html=True)
delta_ph = st.empty()

st.markdown('<div class="section-title">CONFIDENCE  &  DISTANCES</div>', unsafe_allow_html=True)
gc, gd = st.columns(2)
with gc: conf_ph = st.empty()
with gd: dist_ph = st.empty()

st.markdown('<div class="section-title">OPPERVLAKTE</div>', unsafe_allow_html=True)
opp_ph = st.empty()

# ─────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────
def render_images(row):
    img_bgr = row.get("_img_bgr")
    if img_bgr is None:
        raw_ph.image(BLANK, use_container_width=True)
        inf_ph.image(BLANK, use_container_width=True)
        return
    raw_ph.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                 channels="RGB", use_container_width=True)
    inf_ph.image(draw_detections(img_bgr, row),
                 channels="RGB", use_container_width=True)

def render_info(row):
    pf     = row.get("pass_fail")
    tag    = "✅ PASS" if pf == 1 else "❌ FAIL" if pf == 0 else "?"
    color  = "#39ff14" if pf == 1 else "#ff4757"
    reason = f"  ·  {row['fail_reason']}" if row.get("fail_reason") else ""
    sx     = row.get("scale_x_mm_per_px")
    scale  = f"{sx:.4f} mm/px" if sx else "—"
    info_ph.markdown(
        f'<div style="font-family:Share Tech Mono,monospace;font-size:12px;'
        f'color:#6b8099;padding:6px 0 10px 0;">'
        f'File: <span style="color:#00e5ff">{row["id"]}</span>  ·  '
        f'<span style="color:{color}">{tag}</span>{reason}  ·  '
        f'Scale: {scale}</div>',
        unsafe_allow_html=True)

def render_kpis(history: list):
    if not history: return
    df       = pd.DataFrame(history)
    total    = len(df)                                         # all images incl. Ongevuld
    ananas   = df[df.get("ananas_detected", pd.Series(0, index=df.index)) == 1]
    n_ananas = len(ananas)
    passed   = int((ananas["pass_fail"] == 1).sum()) if not ananas.empty else 0
    failed   = int((ananas["pass_fail"] == 0).sum()) if not ananas.empty else 0
    pct      = f"{passed/n_ananas*100:.1f}%" if n_ananas else "—"
    elapsed_min = max(1, (pd.Timestamp.now() - df["triggered_at"].min()).total_seconds() / 60)
    rate     = f"{n_ananas / elapsed_min:.1f}"
    avg_sx   = df["scale_x_mm_per_px"].dropna().mean() if "scale_x_mm_per_px" in df.columns else None
    scale    = f"{avg_sx:.4f}" if avg_sx else "—"
    for ph, label, val in zip(kpi,
            ["Images total","Ananas","Pass ✅","Fail ❌","Pass %","Rate / min"],
            [total, n_ananas, passed, failed, pct, rate]):
        ph.metric(label, val)

def render_graphs(history: list):
    # Drop the raw image from the df — it can't go into pandas columns
    rows = [{k: v for k, v in r.items() if k != "_img_bgr"} for r in history]
    df = add_derived(pd.DataFrame(rows))
    throughput_ph.plotly_chart(graph_throughput(df),                      use_container_width=True)
    scatter_ph.plotly_chart(graph_centering_scatter(df, drill_r, margin), use_container_width=True)
    dmin_ph.plotly_chart(graph_dmin(df, drill_r, margin),                 use_container_width=True)
    delta_ph.plotly_chart(graph_delta_ts(df),                             use_container_width=True)
    conf_ph.plotly_chart(graph_confidence(df),                            use_container_width=True)
    dist_ph.plotly_chart(graph_distances(df),                             use_container_width=True)
    opp_ph.plotly_chart(graph_opp(df),                                    use_container_width=True)


# ─────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────
if "folder_index" not in st.session_state:
    st.session_state.folder_index = 0
if "folder_done" not in st.session_state:
    st.session_state.folder_done = False
if "history" not in st.session_state:
    st.session_state.history = []

# ─────────────────────────────────────────
# MAIN FRAGMENT
# ─────────────────────────────────────────
files = get_image_files(image_folder)

if not files:
    status_ph.markdown(
        f'<div class="status-bar"><span class="warn">✖ No images found in: {image_folder}</span></div>',
        unsafe_allow_html=True)
    st.stop()

model = load_model(model_path)

@st.fragment(run_every=loop_speed)
def inference_loop():
    idx   = st.session_state.folder_index
    total = len(files)

    if st.session_state.folder_done:
        status_ph.markdown(
            f'<div class="status-bar"><span class="idle">■ DONE — {total} images processed</span></div>',
            unsafe_allow_html=True)
        # Still render final state of graphs
        if st.session_state.history:
            render_kpis(st.session_state.history)
            render_graphs(st.session_state.history)
        return

    status_ph.markdown(
        f'<div class="status-bar"><span class="live">▶ PROCESSING</span>'
        f'  <span style="color:#334455;">{idx + 1} / {total}  —  {files[idx].name}</span></div>',
        unsafe_allow_html=True)

    # Load & run inference
    img_bgr = load_image(files[idx])
    if img_bgr is not None:
        detections   = run_inference(model, img_bgr, conf_thresh)
        distances_px = compute_distances_px(detections)
        row          = build_row(files[idx], img_bgr, detections, distances_px)

        st.session_state.history.append(row)

        render_images(row)
        render_info(row)
        render_kpis(st.session_state.history)
        render_graphs(st.session_state.history)

    # Advance
    if idx + 1 >= total:
        st.session_state.folder_done = True
    else:
        st.session_state.folder_index = idx + 1

inference_loop()



