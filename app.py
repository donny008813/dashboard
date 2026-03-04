import streamlit as st
import cv2
from ultralytics import YOLO
import time
import numpy as np
import plotly.graph_objects as go
import os
import pandas as pd

# ---------------- SETTINGS ----------------
VIDEO_PATH = "C:/Users/donny/Documents/Stage/Foto/Video/VID20260206101505.mp4"
MODEL_PATH = "C:/Users/donny/Documents/Stage/Programmas/runs/detect/train7/weights/best.pt"

# Counting zone (ROI)
BOX_X1 = 484
BOX_X2 = 716
BOX_Y1 = 295
BOX_Y2 = 571

CONF_THRESHOLD = 0.8
MAX_RATE = 16
ONLY_CLASS = "Gevuld"

FRAME_SKIP = 6
DISTANCE_INTERVAL = 3.75  # seconden
# ------------------------------------------

st.set_page_config(layout="wide")
st.title("Pineapple Production Monitor")

# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Model file not found:\n{path}")
        st.stop()
    model = YOLO(path)
    model.to("cpu")  # "cuda" als GPU beschikbaar
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

# Track IDs die al geteld zijn
if "counted_ids" not in st.session_state:
    st.session_state.counted_ids = set()

# Afstandsgrafieken
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

# -------- INITIALIZE FILL GRAPH --------
time_array = np.array([0] + st.session_state.time_minutes[1:])
max_fill = np.full_like(time_array, MAX_RATE, dtype=int)
max_fill[0] = 0
cum_auc_max = np.cumsum(max_fill)
cum_auc_actual = np.zeros_like(cum_auc_max)
cum_percentage = np.zeros_like(cum_auc_max)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=time_array,
    y=max_fill,
    mode='lines+markers',
    name="Max Fill Rate",
    customdata=np.stack([cum_auc_max, np.full_like(cum_auc_max, 100)], axis=-1),
    hovertemplate='Time: %{x} min<br>Rate: %{y} units/min<br>Cumulative AUC: %{customdata[0]}<br>Cumulative %: %{customdata[1]:.1f}%<extra></extra>'
))
fig.add_trace(go.Scatter(
    x=time_array,
    y=cum_auc_actual,
    mode='lines+markers',
    name="Actual Fill per Minute",
    customdata=np.stack([cum_auc_actual, cum_percentage], axis=-1),
    hovertemplate='Time: %{x} min<br>Rate: %{y} units/min<br>Cumulative AUC: %{customdata[0]}<br>Cumulative %: %{customdata[1]:.1f}%<extra></extra>'
))
fig.update_layout(
    title="Max Fill vs Actual Fill per Minute",
    xaxis_title="Time (minutes)",
    yaxis_title="Units per Minute",
    hovermode="x unified"
)
chart_placeholder.plotly_chart(fig, use_container_width=True)

# -------- HELPER FUNCTION TO UPDATE FILL GRAPH --------
def update_graph():
    time_array = np.array([0] + st.session_state.time_minutes[1:])
    max_fill = np.full_like(time_array, MAX_RATE, dtype=int)
    max_fill[0] = 0

    sim_fill = np.array([0] + st.session_state.per_minute_counts)
    if len(sim_fill) < len(max_fill):
        sim_fill = np.pad(sim_fill, (0, len(max_fill)-len(sim_fill)), 'constant')

    cum_auc_max = np.cumsum(max_fill)
    cum_auc_sim = np.cumsum(sim_fill)
    cum_percentage = (cum_auc_sim / cum_auc_max) * 100

    min_len = min(len(cum_auc_sim), len(cum_percentage), len(max_fill))
    fig.data[0].x = time_array[:min_len]
    fig.data[0].y = max_fill[:min_len]
    fig.data[0].customdata = np.stack([cum_auc_max[:min_len], np.full(min_len, 100)], axis=-1)

    fig.data[1].x = time_array[:min_len]
    fig.data[1].y = sim_fill[:min_len]
    fig.data[1].customdata = np.stack([cum_auc_sim[:min_len], cum_percentage[:min_len]], axis=-1)

    chart_placeholder.plotly_chart(fig, use_container_width=True)

# -------- UPDATE DISTANCE GRAFIEKEN --------
def update_distance_plots():
    if st.session_state.distance_time:
        # ΔX
        fig_x = go.Figure()
        fig_x.add_trace(go.Scatter(
            x=st.session_state.distance_time,
            y=st.session_state.delta_x,
            mode='lines+markers',
            name="ΔX (hor.)"
        ))
        fig_x.update_layout(
            title="Horizontale offset Ananas → Plateau",
            xaxis_title="Time (s)",
            yaxis_title="ΔX (pixels)"
        )
        distance_x_placeholder.plotly_chart(fig_x, use_container_width=True)

        # ΔY
        fig_y = go.Figure()
        fig_y.add_trace(go.Scatter(
            x=st.session_state.distance_time,
            y=st.session_state.delta_y,
            mode='lines+markers',
            name="ΔY (vert.)"
        ))
        fig_y.update_layout(
            title="Verticale offset Ananas → Plateau",
            xaxis_title="Time (s)",
            yaxis_title="ΔY (pixels)"
        )
        distance_y_placeholder.plotly_chart(fig_y, use_container_width=True)

# -------- AFSTAND FUNCTION --------
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

# -------- MAIN LOOP --------
if st.session_state.running:
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        st.error("Could not open video file.")
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

                # Counting logic based on track_id
                if BOX_X1 <= cx <= BOX_X2 and BOX_Y1 <= cy <= BOX_Y2:
                    if track_id not in st.session_state.counted_ids:
                        st.session_state.minute_counter += 1
                        st.session_state.counted_ids.add(track_id)

                # Draw bounding box
                label = f"{class_name} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # ----- DISTANCE UPDATE EVERY 3.75s -----
            current_time = time.time() - st.session_state.start_time
            if current_time - st.session_state.last_distance_time >= DISTANCE_INTERVAL:
                delta_x, delta_y = compute_center_deltas(boxes, classes)
                if delta_x is not None:
                    st.session_state.distance_time.append(current_time)
                    st.session_state.delta_x.append(delta_x)
                    st.session_state.delta_y.append(delta_y)
                    update_distance_plots()
                st.session_state.last_distance_time = current_time

        # Draw counting ROI
        cv2.rectangle(frame, (BOX_X1, BOX_Y1), (BOX_X2, BOX_Y2), (255, 0, 0), 3)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB")

        # --- Update per-minute stats & graph ---
        elapsed_minutes = int((time.time() - st.session_state.start_time) / 60)
        if elapsed_minutes >= len(st.session_state.per_minute_counts) + 1:
            st.session_state.per_minute_counts.append(st.session_state.minute_counter)
            st.session_state.minute_counter = 0
            st.session_state.time_minutes.append(elapsed_minutes)
            update_graph()

        # --- Metrics ---
        cumulative_fill = sum(st.session_state.per_minute_counts) + st.session_state.minute_counter
        total_minutes_elapsed = max(1, len(st.session_state.per_minute_counts) + 1)
        max_possible_total = MAX_RATE * total_minutes_elapsed
        cumulative_percentage = (cumulative_fill / max_possible_total) * 100

        with count_placeholder.container():
            col1, col2, col3 = st.columns(3)
            col1.metric("Units this minute", st.session_state.minute_counter)
            col2.metric("Cumulative Fill", cumulative_fill)
            col3.metric("Cumulative %", f"{cumulative_percentage:.1f}%")

        time.sleep(0.01)

    cap.release()

# ================================
# SIMPELE SIMULATIE GRAFIEKEN
# ================================

st.subheader("Simulatie: Centrering Analyse")

# -----------------------
# Data één keer genereren
# -----------------------
if "sim_data" not in st.session_state:
    np.random.seed(42)
    n_products = 60
    D_min = np.random.normal(loc=37, scale=7, size=n_products)
    delta_x = np.random.normal(loc=0, scale=5, size=n_products)
    delta_y = np.random.normal(loc=0, scale=5, size=n_products)

    st.session_state.sim_data = pd.DataFrame({
        "tijd": np.arange(1, n_products+1),
        "D_min": D_min,
        "delta_x": delta_x,
        "delta_y": delta_y
    })

df = st.session_state.sim_data

# -----------------------
# Layout: dropdowns links, grafieken rechts
# -----------------------
col_settings, col_graph = st.columns([1, 3])

with col_settings:
    st.subheader("Instellingen")
    r_b = st.selectbox("Kies boorstraal (mm)", [25, 30, 35, 40], index=1)
    marge = st.selectbox("Kies marge (mm)", [3, 5, 7, 10], index=1)

# -----------------------
# Classificatie op basis van dropdowns
# -----------------------
conditions = [
    df["D_min"] >= r_b + marge,
    (df["D_min"] >= r_b) & (df["D_min"] < r_b + marge),
    df["D_min"] < r_b
]
choices = ["Goed", "Risicozone", "Fout"]
df["klasse"] = np.select(conditions, choices, default="Onbekend")

colors = {"Goed": "green", "Risicozone": "orange", "Fout": "red", "Onbekend": "gray"}

# -----------------------
# Grafieken rechts
# -----------------------
with col_graph:
    # 1️⃣ Scatterplot Δx vs Δy
    fig1 = go.Figure()
    for label, group in df.groupby("klasse"):
        fig1.add_trace(go.Scatter(
            x=group["delta_x"],
            y=group["delta_y"],
            mode="markers",
            name=label,
            marker=dict(size=10, color=colors[label], opacity=0.7)
        ))

    fig1.add_hline(y=0, line_dash="dash", line_color="black")
    fig1.add_vline(x=0, line_dash="dash", line_color="black")

    fig1.update_layout(
        title="Centrering kern t.o.v. ananas (Δx vs Δy)",
        xaxis_title="Δx (mm)",
        yaxis_title="Δy (mm)",
        yaxis_scaleanchor="x",
        height=400
    )

    st.plotly_chart(fig1, use_container_width=True)

    # 2️⃣ Tijdreeks D_min
    fig2 = go.Figure()
    for label, group in df.groupby("klasse"):
        fig2.add_trace(go.Scatter(
            x=group["tijd"],
            y=group["D_min"],
            mode="markers",
            name=label,
            marker=dict(size=9, color=colors[label])
        ))

    # Verbindende lijn
    fig2.add_trace(go.Scatter(
        x=df["tijd"],
        y=df["D_min"],
        mode="lines",
        line=dict(color="blue", width=2),
        opacity=0.3,
        showlegend=False
    ))

    # Referentielijnen
    fig2.add_hline(y=r_b, line_dash="dash", line_color="red", annotation_text="Boorstraal")
    fig2.add_hline(y=r_b + marge, line_dash="dash", line_color="orange", annotation_text="Boorstraal + marge")

    fig2.update_layout(
        title="Tijdreeks van D_min per product",
        xaxis_title="Product / Tijd",
        yaxis_title="Minimale afstand tot ananas rand (D_min)",
        height=400
    )

    fig2.update_yaxes(autorange="reversed")

    st.plotly_chart(fig2, use_container_width=True)

###################
st.subheader("Simulatie: plateau positionering")

if "sim_data_positie" not in st.session_state:
    np.random.seed(42)
    n_products = 60

    # Basisdata
    D_min = np.random.normal(loc=37, scale=7, size=n_products)
    delta_x_positie = np.random.normal(loc=0, scale=5, size=n_products)
    delta_y_positie = np.random.normal(loc=0, scale=5, size=n_products)

    # -----------------------
    # delta_x_lineair simulatie
    # -----------------------
    n_rond_nul = 30        # aantal eerste tijdstappen rond 0
    delta_x_end = 20       # eindwaarde lineaire trend
    noise_std = 5          # ruis tijdens lineaire fase
    delta_x_lineair = np.zeros(n_products)

    # Eerste n_rond_nul stappen: normaal rond 0
    delta_x_lineair[:n_rond_nul] = np.random.normal(loc=0, scale=5, size=n_rond_nul)

    # Lineair stijgend met ruis vanaf stap n_rond_nul
    t_linear = np.arange(n_rond_nul, n_products)
    n_linear = len(t_linear)
    linear_trend = np.linspace(0, delta_x_end, n_linear)
    delta_x_lineair[n_rond_nul:] = linear_trend + np.random.normal(0, noise_std, size=n_linear)

    # -----------------------
    # DataFrame
    # -----------------------
    st.session_state.sim_data_positie = pd.DataFrame({
        "tijd": np.arange(1, n_products+1),
        "D_min": D_min,
        "delta_x": delta_x_positie,
        "delta_y": delta_y_positie,
        "delta_x_lineair": delta_x_lineair
    })

df_positie = st.session_state.sim_data_positie

y_std = df["delta_y"].std()

variant = st.selectbox(
    "Selecteer variant van de positionering grafiek:",
    options=["delta_x", "delta_x_lineair"],
    index=0
)

fig3 = go.Figure()

for x in df_positie:
    fig3.add_trace(go.Scatter(
        x=df_positie["tijd"],
        y=df_positie[variant],
        mode="markers",
        marker=dict(size=9, color="blue"),
        showlegend=False
        ))

# Lijnen tussen punten
fig3.add_trace(go.Scatter(
    x=df_positie["tijd"],
    y=df_positie[variant],
    mode="lines",
    line=dict(color="black", width=2),
    opacity=0.3,
    showlegend=False
))

# Referentie lijn std_y positief
fig3.add_hline(
    y= 2 * y_std,
    line_dash="dash",
    line_color="black"
)

# Referentie lijn std_y negatief
fig3.add_hline(
    y= -(2 * y_std),
    line_dash="dash",
    line_color="black"
)

st.plotly_chart(fig3)