"""
LAB MONITORING CLOUD AI - 3 SENSOR (MQ135, MQ2, MQ7)
Streamlit Dashboard with ML Prediction
"""

import json
import time
import queue
import uuid
from datetime import datetime, timedelta, timezone

import streamlit as st
import pandas as pd
import joblib
import altair as alt
import paho.mqtt.client as mqtt

# --- Configuration ---
BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC_DATA = "net4think/lab_monitor/data"
TOPIC_PRED_MQ135 = "net4think/lab_monitor/pred_mq135"
TOPIC_PRED_MQ2 = "net4think/lab_monitor/pred_mq2"
TOPIC_PRED_MQ7 = "net4think/lab_monitor/pred_mq7"

# Model Files
MODEL_MQ135 = "air_quality_rf_model.joblib"
MODEL_MQ2 = "model_mq2.joblib"
MODEL_MQ7 = "model_mq7.joblib"

st.set_page_config(
    page_title="Lab Monitor AI",
    page_icon="🧪",
    layout="wide"
)

# --- State Management ---
if 'mqtt_queue' not in st.session_state:
    st.session_state.mqtt_queue = queue.Queue()

if 'sensor_data' not in st.session_state:
    st.session_state.sensor_data = {
        "temp": 0, "hum": 0, 
        "mq135": 0, "mq2": 0, "mq7": 0,
        "timestamp": "-"
    }

if 'predictions' not in st.session_state:
    st.session_state.predictions = {
        "mq135": {"label": "Menunggu...", "confidence": 0},
        "mq2": {"label": "Menunggu...", "confidence": 0},
        "mq7": {"label": "Menunggu...", "confidence": 0}
    }

if 'history' not in st.session_state:
    st.session_state.history = []

# --- Model Loading ---
@st.cache_resource
def load_models():
    """Load all ML models"""
    models = {}
    
    # MQ-135 Model (with label encoder)
    try:
        artifact = joblib.load(MODEL_MQ135)
        models['mq135'] = {
            'model': artifact["model"],
            'label_encoder': artifact["label_encoder"],
            'features': artifact["features"]
        }
        print("✅ MQ-135 model loaded")
    except Exception as e:
        print(f"❌ MQ-135 model error: {e}")
        models['mq135'] = None
    
    # MQ-2 Model (simple RF)
    try:
        models['mq2'] = joblib.load(MODEL_MQ2)
        print("✅ MQ-2 model loaded")
    except Exception as e:
        print(f"❌ MQ-2 model error: {e}")
        models['mq2'] = None
    
    # MQ-7 Model (simple RF)
    try:
        models['mq7'] = joblib.load(MODEL_MQ7)
        print("✅ MQ-7 model loaded")
    except Exception as e:
        print(f"❌ MQ-7 model error: {e}")
        models['mq7'] = None
    
    return models

models = load_models()

# --- MQTT Setup ---
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("Connected to MQTT Broker!")
        client.subscribe(TOPIC_DATA)
    else:
        print(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        if userdata is not None:
            userdata.put((msg.topic, payload))
    except Exception as e:
        print(f"MQTT Error: {e}")

@st.cache_resource
def start_mqtt():
    unique_id = f"Streamlit_Lab_{uuid.uuid4().hex[:8]}"
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=unique_id, clean_session=True)
    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect(BROKER, PORT, 60)
        client.loop_start()
        return client
    except Exception as e:
        st.error(f"MQTT Connection Error: {e}")
        return None

mqtt_client = start_mqtt()
if mqtt_client:
    mqtt_client.user_data_set(st.session_state.mqtt_queue)

# --- Prediction Functions ---
def predict_mq135(temp, hum, gas):
    """Predict air quality using MQ-135 model"""
    if models['mq135'] is None:
        return "N/A", 0
    
    try:
        m = models['mq135']
        input_df = pd.DataFrame([[temp, hum, gas]], columns=m['features'])
        pred_idx = m['model'].predict(input_df)[0]
        label = m['label_encoder'].inverse_transform([pred_idx])[0]
        proba = m['model'].predict_proba(input_df)[0]
        confidence = round(proba[pred_idx] * 100, 1)
        return label, confidence
    except Exception as e:
        print(f"MQ135 prediction error: {e}")
        return "Error", 0

def predict_mq2(mq2_ppm):
    """Predict smoke using MQ-2 model"""
    if models['mq2'] is None:
        return "N/A", 0
    
    try:
        pred = models['mq2'].predict([[mq2_ppm]])[0]
        proba = models['mq2'].predict_proba([[mq2_ppm]])[0]
        confidence = round(max(proba) * 100, 1)
        return pred, confidence
    except Exception as e:
        print(f"MQ2 prediction error: {e}")
        return "Error", 0

def predict_mq7(mq7_ppm):
    """Predict CO/gas using MQ-7 model"""
    if models['mq7'] is None:
        return "N/A", 0
    
    try:
        pred = models['mq7'].predict([[mq7_ppm]])[0]
        proba = models['mq7'].predict_proba([[mq7_ppm]])[0]
        confidence = round(max(proba) * 100, 1)
        return pred, confidence
    except Exception as e:
        print(f"MQ7 prediction error: {e}")
        return "Error", 0

# --- Data Processing ---
while not st.session_state.mqtt_queue.empty():
    topic, payload = st.session_state.mqtt_queue.get()
    
    if topic == TOPIC_DATA:
        # Parse Data
        temp = float(payload.get("temperature", 0))
        hum = float(payload.get("humidity", 0))
        mq135 = float(payload.get("mq135_ppm", 0))
        mq2 = float(payload.get("mq2_ppm", 0))
        mq7 = float(payload.get("mq7_ppm", 0))
        
        wib_time = datetime.now(timezone.utc) + timedelta(hours=7)
        timestamp = wib_time.strftime("%H:%M:%S")

        # Update State
        st.session_state.sensor_data = {
            "temp": temp, "hum": hum,
            "mq135": mq135, "mq2": mq2, "mq7": mq7,
            "timestamp": timestamp
        }
        
        # Predictions
        label_mq135, conf_mq135 = predict_mq135(temp, hum, mq135)
        label_mq2, conf_mq2 = predict_mq2(mq2)
        label_mq7, conf_mq7 = predict_mq7(mq7)
        
        st.session_state.predictions = {
            "mq135": {"label": label_mq135, "confidence": conf_mq135},
            "mq2": {"label": label_mq2, "confidence": conf_mq2},
            "mq7": {"label": label_mq7, "confidence": conf_mq7}
        }

        # Publish Predictions to MQTT
        if mqtt_client:
            mqtt_client.publish(TOPIC_PRED_MQ135, json.dumps({
                "label": label_mq135, "confidence": conf_mq135
            }))
            mqtt_client.publish(TOPIC_PRED_MQ2, json.dumps({
                "label": label_mq2, "confidence": conf_mq2
            }))
            mqtt_client.publish(TOPIC_PRED_MQ7, json.dumps({
                "label": label_mq7, "confidence": conf_mq7
            }))
        
        # Update History
        new_record = {
            "time": timestamp,
            "Temp": temp, "Hum": hum,
            "MQ135": mq135, "MQ2": mq2, "MQ7": mq7,
            "Status_135": label_mq135,
            "Status_MQ2": label_mq2,
            "Status_MQ7": label_mq7
        }
        st.session_state.history.append(new_record)
        
        if len(st.session_state.history) > 1000:
            st.session_state.history.pop(0)

# --- UI Styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
.main-header { font-size: 2.5rem; font-weight: 600; color: #1e3a8a; text-align: center; }
.metric-card {
    background: white; padding: 20px; border-radius: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05); text-align: center;
    border: 1px solid #e2e8f0; margin-bottom: 10px;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #0f172a; }
.metric-label { font-size: 0.9rem; color: #64748b; text-transform: uppercase; }
.sensor-card {
    padding: 20px; border-radius: 16px; color: white; text-align: center;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15); margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="main-header">🧪 Lab Monitoring AI Dashboard</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#64748b;">Real-time Chemical Lab Safety Monitoring with 3 ML Models</p>', unsafe_allow_html=True)

# --- Sensor Status Cards ---
st.subheader("📊 Sensor Predictions")

col1, col2, col3 = st.columns(3)

# MQ-135 Card
with col1:
    p = st.session_state.predictions['mq135']
    color = "#10b981" if p['label'] == "Baik" else "#f59e0b" if p['label'] == "Sedang" else "#ef4444"
    st.markdown(f"""
    <div class="sensor-card" style="background: linear-gradient(135deg, {color}, {color}cc);">
        <h4 style="margin:0; opacity:0.9;">💨 MQ-135 (Air Quality)</h4>
        <h2 style="margin:10px 0; font-size:2rem;">{p['label'].replace("_", " ")}</h2>
        <p style="margin:0;">Confidence: {p['confidence']}%</p>
        <p style="margin:5px 0 0 0; font-size:1.5rem;">{st.session_state.sensor_data['mq135']:.1f} ppm</p>
    </div>
    """, unsafe_allow_html=True)

# MQ-2 Card
with col2:
    p = st.session_state.predictions['mq2']
    color = "#ef4444" if p['label'] == "smoke" else "#10b981"
    icon = "🔥" if p['label'] == "smoke" else "✅"
    st.markdown(f"""
    <div class="sensor-card" style="background: linear-gradient(135deg, {color}, {color}cc);">
        <h4 style="margin:0; opacity:0.9;">{icon} MQ-2 (Smoke)</h4>
        <h2 style="margin:10px 0; font-size:2rem;">{p['label'].upper()}</h2>
        <p style="margin:0;">Confidence: {p['confidence']}%</p>
        <p style="margin:5px 0 0 0; font-size:1.5rem;">{st.session_state.sensor_data['mq2']:.1f} ppm</p>
    </div>
    """, unsafe_allow_html=True)

# MQ-7 Card
with col3:
    p = st.session_state.predictions['mq7']
    color = "#ef4444" if p['label'] == "smoke" else "#10b981"
    icon = "☠️" if p['label'] == "smoke" else "✅"
    st.markdown(f"""
    <div class="sensor-card" style="background: linear-gradient(135deg, {color}, {color}cc);">
        <h4 style="margin:0; opacity:0.9;">{icon} MQ-7 (CO/Gas)</h4>
        <h2 style="margin:10px 0; font-size:2rem;">{p['label'].upper()}</h2>
        <p style="margin:0;">Confidence: {p['confidence']}%</p>
        <p style="margin:5px 0 0 0; font-size:1.5rem;">{st.session_state.sensor_data['mq7']:.1f} ppm</p>
    </div>
    """, unsafe_allow_html=True)

# --- Environment Metrics ---
st.subheader("🌡️ Environment")
c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Temperature", f"{st.session_state.sensor_data['temp']:.1f} °C")
with c2:
    st.metric("Humidity", f"{st.session_state.sensor_data['hum']:.1f} %")
with c3:
    st.metric("Last Update", st.session_state.sensor_data['timestamp'])

# --- Charts ---
if st.session_state.history:
    st.subheader("📈 Trend Analysis")
    
    df = pd.DataFrame(st.session_state.history)
    df = df.reset_index(names='step')
    
    tab1, tab2 = st.tabs(["💨 Gas Sensors", "🌡️ Environment"])
    
    with tab1:
        # Multi-line chart for all gas sensors
        df_melt = df.melt(id_vars=['step', 'time'], value_vars=['MQ135', 'MQ2', 'MQ7'],
                         var_name='Sensor', value_name='PPM')
        
        chart = alt.Chart(df_melt).mark_line(interpolate='monotone', strokeWidth=2).encode(
            x=alt.X('step:Q', title='Time Steps'),
            y=alt.Y('PPM:Q', title='Gas Concentration (ppm)'),
            color=alt.Color('Sensor:N', scale=alt.Scale(
                domain=['MQ135', 'MQ2', 'MQ7'],
                range=['#3b82f6', '#ef4444', '#f59e0b']
            )),
            tooltip=['time', 'Sensor', 'PPM']
        ).properties(height=350).interactive()
        
        st.altair_chart(chart, use_container_width=True)
    
    with tab2:
        base = alt.Chart(df).encode(x=alt.X('step:Q', title='Time Steps'))
        
        line_temp = base.mark_line(color='#ef4444', strokeWidth=2).encode(
            y=alt.Y('Temp:Q', title='Temperature (°C)')
        )
        line_hum = base.mark_line(color='#3b82f6', strokeWidth=2).encode(
            y=alt.Y('Hum:Q', title='Humidity (%)')
        )
        
        chart = alt.layer(line_temp, line_hum).resolve_scale(y='independent').properties(height=350).interactive()
        st.altair_chart(chart, use_container_width=True)
    
    # Data Table
    with st.expander("📄 Raw Data"):
        st.dataframe(df[['time', 'Temp', 'Hum', 'MQ135', 'MQ2', 'MQ7', 'Status_135', 'Status_MQ2', 'Status_MQ7']], 
                    use_container_width=True, hide_index=True)
        
        st.download_button(
            "📥 Download CSV",
            df.to_csv(index=False).encode('utf-8'),
            f"lab_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )
else:
    st.info("⏳ Waiting for sensor data...")

# Auto-refresh
time.sleep(2)
st.rerun()
