import streamlit as st
import cv2
from ultralytics import YOLO
import time
import numpy as np
import plotly.graph_objects as go
import os
import pandas as pd

# ---------------- SETTINGS ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_PATH = os.path.join(BASE_DIR, "assets", "fixed_video.mp4")
MODEL_PATH = os.path.join(BASE_DIR, "assets", "train7", "weights", "best.pt")

BOX_X1 = 484
BOX_X2 = 716
BOX_Y1 = 295
BOX_Y2 = 571

CONF_THRESHOLD = 0.8
MAX_RATE = 16
ONLY_CLASS = "Gevuld"

FRAME_SKIP = 6
DISTANCE_INTERVAL = 3.75

# ------------------------------------------

st.set_page_config(layout="wide")
st.title("Pineapple Production Monitor")

# ---------------- FILE CHECKS ----------------

if not os.path.exists(VIDEO_PATH):
    st.error(f"Video not found: {VIDEO_PATH}")
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found: {MODEL_PATH}")
    st.stop()

# ---------------- MODEL LOADING ----------------

@st.cache_resource
def load_model(path):
    model = YOLO(path)
    model.to("cpu")
    return model

model = load_model(MODEL_PATH)

# -------- SESSION STATE INIT --------

if "running" not in st.session_state:
    st.session_state.running = False

if "minute_counter" not in st.session_state:
    st.session_state.minute_counter = 0
    st.session_state.per_minute_counts = []
    st.session_state.time_minutes = [0]
    st.session_state.start_time = time.time()

if "counted_ids" not in st.session_state:
    st.session_state.counted_ids = set()

if "distance_time" not in st.session_state:
    st.session_state.distance_time = []
    st.session_state.delta_x = []
    st.session_state.delta_y = []
    st.session_state.last_distance_time = 0

# -------- CONTROLS --------

col1, col2 = st.columns(2)

if col1.button("▶ Start"):
    st.session_state.running = True
    st.session_state.start_time = time.time()
    st.session_state.counted_ids = set()
    st.session_state.minute_counter = 0
    st.session_state.per_minute_counts = []
    st.session_state.time_minutes = [0]
    st.session_state.distance_time = []
    st.session_state.delta_x = []
    st.session_state.delta_y = []
    st.session_state.last_distance_time = 0

if col2.button("⏹ Stop"):
    st.session_state.running = False

video_placeholder = st.empty()
chart_placeholder = st.empty()
distance_x_placeholder = st.empty()
distance_y_placeholder = st.empty()
count_placeholder = st.empty()

# -------- INITIAL GRAPH --------

fig = go.Figure()
fig.update_layout(
    title="Max Fill vs Actual Fill per Minute",
    xaxis_title="Time (minutes)",
    yaxis_title="Units per Minute",
    hovermode="x unified"
)

chart_placeholder.plotly_chart(fig, use_container_width=True)

# -------- DISTANCE FUNCTION --------

def compute_center_deltas(boxes, classes):

    pineapple_boxes = []
    plateau_boxes = []

    for box, cls in zip(boxes, classes):

        class_name = model.names[int(cls)].lower()

        if class_name == "ananas":
            pineapple_boxes.append(box)

        elif class_name == "plateau":
            plateau_boxes.append(box)

    if not pineapple_boxes or not plateau_boxes:
        return None, None

    x1_a, y1_a, x2_a, y2_a = pineapple_boxes[0]
    x1_p, y1_p, x2_p, y2_p = plateau_boxes[0]

    delta_x = ((x1_a + x2_a)/2) - ((x1_p + x2_p)/2)
    delta_y = ((y1_a + y2_a)/2) - ((y1_p + y2_p)/2)

    return delta_x, delta_y

# -------- VIDEO LOOP --------

if st.session_state.running:

    cap = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        st.error("Could not open video.")
        st.stop()

    frame_count = 0
    last_results = None

    while cap.isOpened() and st.session_state.running:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        if frame_count % FRAME_SKIP == 0:
            last_results = model.track(frame, persist=True, conf=CONF_THRESHOLD)

        if last_results and last_results[0].boxes.id is not None:

            boxes = last_results[0].boxes.xyxy.cpu().numpy()
            ids = last_results[0].boxes.id.cpu().numpy()
            classes = last_results[0].boxes.cls.cpu().numpy()
            confs = last_results[0].boxes.conf.cpu().numpy()

            for box, track_id, cls, conf in zip(boxes, ids, classes, confs):

                class_name = model.names[int(cls)]

                if ONLY_CLASS and class_name != ONLY_CLASS:
                    continue

                x1, y1, x2, y2 = map(int, box)

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                if BOX_X1 <= cx <= BOX_X2 and BOX_Y1 <= cy <= BOX_Y2:

                    if track_id not in st.session_state.counted_ids:
                        st.session_state.minute_counter += 1
                        st.session_state.counted_ids.add(track_id)

                label = f"{class_name} {conf:.2f}"

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)

                cv2.putText(frame,label,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        cv2.rectangle(frame,(BOX_X1,BOX_Y1),(BOX_X2,BOX_Y2),(255,0,0),3)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB")

        cumulative_fill = sum(st.session_state.per_minute_counts) + st.session_state.minute_counter

        with count_placeholder.container():

            c1, c2 = st.columns(2)

            c1.metric("Units this minute", st.session_state.minute_counter)
            c2.metric("Total counted", cumulative_fill)

        time.sleep(0.03)

    cap.release()

# =====================================================
# SIMULATION: CENTERING ANALYSIS
# =====================================================

st.subheader("Simulatie: Centrering Analyse")

if "sim_data" not in st.session_state:

    np.random.seed(42)

    n_products = 60

    D_min = np.random.normal(37,7,n_products)
    delta_x = np.random.normal(0,5,n_products)
    delta_y = np.random.normal(0,5,n_products)

    st.session_state.sim_data = pd.DataFrame({
        "tijd": np.arange(1,n_products+1),
        "D_min": D_min,
        "delta_x": delta_x,
        "delta_y": delta_y
    })

df = st.session_state.sim_data

col_settings, col_graph = st.columns([1,3])

with col_settings:

    st.subheader("Instellingen")

    r_b = st.selectbox("Kies boorstraal (mm)", [25,30,35,40], index=1)
    marge = st.selectbox("Kies marge (mm)", [3,5,7,10], index=1)

conditions = [
    df["D_min"] >= r_b + marge,
    (df["D_min"] >= r_b) & (df["D_min"] < r_b + marge),
    df["D_min"] < r_b
]

choices = ["Goed","Risicozone","Fout"]

df["klasse"] = np.select(conditions, choices, default="Onbekend")

colors = {
    "Goed":"green",
    "Risicozone":"orange",
    "Fout":"red",
    "Onbekend":"gray"
}

with col_graph:

    fig1 = go.Figure()

    for label, group in df.groupby("klasse"):

        fig1.add_trace(go.Scatter(
            x=group["delta_x"],
            y=group["delta_y"],
            mode="markers",
            name=label,
            marker=dict(size=10,color=colors[label],opacity=0.7)
        ))

    fig1.add_hline(y=0,line_dash="dash",line_color="black")
    fig1.add_vline(x=0,line_dash="dash",line_color="black")

    fig1.update_layout(
        title="Centrering kern t.o.v. ananas (Δx vs Δy)",
        xaxis_title="Δx (mm)",
        yaxis_title="Δy (mm)",
        yaxis_scaleanchor="x",
        height=400
    )

    st.plotly_chart(fig1,use_container_width=True)

    fig2 = go.Figure()

    for label, group in df.groupby("klasse"):

        fig2.add_trace(go.Scatter(
            x=group["tijd"],
            y=group["D_min"],
            mode="markers",
            name=label,
            marker=dict(size=9,color=colors[label])
        ))

    fig2.add_trace(go.Scatter(
        x=df["tijd"],
        y=df["D_min"],
        mode="lines",
        line=dict(color="blue",width=2),
        opacity=0.3,
        showlegend=False
    ))

    fig2.add_hline(y=r_b,line_dash="dash",line_color="red")
    fig2.add_hline(y=r_b+marge,line_dash="dash",line_color="orange")

    fig2.update_layout(
        title="Tijdreeks van D_min per product",
        xaxis_title="Product / Tijd",
        yaxis_title="D_min",
        height=400
    )

    fig2.update_yaxes(autorange="reversed")

    st.plotly_chart(fig2,use_container_width=True)

# =====================================================
# SIMULATION: PLATEAU POSITIONING
# =====================================================

st.subheader("Simulatie: plateau positionering")

if "sim_data_positie" not in st.session_state:

    np.random.seed(42)

    n_products = 60

    delta_x = np.random.normal(0,5,n_products)
    delta_y = np.random.normal(0,5,n_products)

    delta_x_lineair = np.linspace(0,20,n_products) + np.random.normal(0,5,n_products)

    st.session_state.sim_data_positie = pd.DataFrame({
        "tijd": np.arange(1,n_products+1),
        "delta_x": delta_x,
        "delta_y": delta_y,
        "delta_x_lineair": delta_x_lineair
    })

df_positie = st.session_state.sim_data_positie

variant = st.selectbox(
    "Selecteer variant van de positionering grafiek:",
    ["delta_x","delta_x_lineair"]
)

fig3 = go.Figure()

fig3.add_trace(go.Scatter(
    x=df_positie["tijd"],
    y=df_positie[variant],
    mode="markers",
    marker=dict(size=9,color="blue")
))

fig3.add_trace(go.Scatter(
    x=df_positie["tijd"],
    y=df_positie[variant],
    mode="lines",
    line=dict(color="black",width=2),
    opacity=0.3
))

y_std = df["delta_y"].std()

fig3.add_hline(y=2*y_std,line_dash="dash")
fig3.add_hline(y=-2*y_std,line_dash="dash")

fig3.update_layout(
    title="Plateau positionering analyse",
    xaxis_title="Product / Tijd",
    yaxis_title="Δx"
)

st.plotly_chart(fig3)




