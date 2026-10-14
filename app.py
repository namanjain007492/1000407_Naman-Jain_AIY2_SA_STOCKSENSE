import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from PIL import Image
from ultralytics import YOLO
from datetime import datetime, timedelta

# 1. UI Configuration & Custom CSS
st.set_page_config(page_title="StockSense Pro | Command Center", page_icon="🛒", layout="wide")
st.markdown("""
    <style>
    .main {background-color: #0E1117;}
    h1, h2, h3 {color: #00FFCC;}
    .stMetric {background-color: #1E2127; padding: 15px; border-radius: 10px; border-left: 5px solid #00FFCC;}
    </style>
""", unsafe_allow_html=True)

st.title("🛒 StockSense Pro // AI TELEMETRY COMMAND CENTER")
st.caption(f"System Online | Local Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Environment: Production")

# 2. Model Initialization
@st.cache_resource
def load_vision_model():
    return YOLO("best.pt")

try:
    model = load_vision_model()
except:
    st.error("SYSTEM HALT: 'best.pt' weights not detected in the root directory.")
    st.stop()

# 3. Multi-Tab Architecture
tab1, tab2, tab3, tab4 = st.tabs(["📷 Live Scanner", "📈 Predictive Analytics", "🗄️ System Logs", "⚙️ Engine Settings"])

with tab4:
    st.subheader("Engine Settings")
    conf_threshold = st.slider("AI Confidence Threshold", 0.1, 1.0, 0.45, 0.05)
    iou_threshold = st.slider("NMS IoU Threshold (Overlap)", 0.1, 1.0, 0.5, 0.05)
    st.markdown("Adjusting these parameters forces the AI to be more or less strict on object boundaries.")

with tab1:
    st.subheader("Real-Time Vision Interface")
    col_input, col_output = st.columns([1, 1.2])
    
    with col_input:
        input_method = st.radio("Select Feed Source:", ["Upload Static Image", "Live Camera Feed"], horizontal=True)
        img_source = None
        
        if input_method == "Upload Static Image":
            img_source = st.file_uploader("Drop high-res shelf scan here", type=["jpg", "png", "jpeg"])
        else:
            img_source = st.camera_input("Initialize Camera")
            
    if img_source:
        image = Image.open(img_source).convert("RGB")
        
        with st.spinner("Processing neural network telemetry..."):
            res = model.predict(image, conf=conf_threshold, iou=iou_threshold)[0]
            
        with col_output:
            st.image(res.plot(), use_container_width=True, caption="YOLOv8 Bounding Box Render")
            
        # Extract Data
        names = model.names
        detected = [names[int(b.cls[0])] for b in res.boxes]
        inventory = {item: detected.count(item) for item in names.values()}
        
        # 4. Live Metrics Engine
        df = pd.DataFrame(list(inventory.items()), columns=["Product", "Qty"])
        df["Status"] = np.where(df["Qty"] == 0, "Critical", np.where(df["Qty"] <= 3, "Warning", "Optimal"))
        
        st.markdown("### ⚡ Live Telemetry Readout")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Objects Tracked", len(detected))
        m2.metric("Critical Shortages", len(df[df['Status']=='Critical']))
        m3.metric("Restock Warnings", len(df[df['Status']=='Warning']))
        m4.metric("Network Inference Time", f"{round(res.speed['inference'], 2)} ms")
        
        # 5. Styled Dataframe
        def color_status(val):
            color = '#ff4b4b' if val == 'Critical' else '#ffa500' if val == 'Warning' else '#00FFCC'
            return f'color: {color}; font-weight: bold;'
            
        st.dataframe(df.style.map(color_status, subset=['Status']), use_container_width=True)

with tab2:
    st.subheader("Predictive Restock Analytics (Simulated)")
    st.markdown("Forecasts generated using historical inventory depletion rates.")
    
    # 6. Simulated Historical Data for Forecasting
    dates = [datetime.now() - timedelta(days=i) for i in range(7)]
    hist_data = pd.DataFrame({
        "Date": dates[::-1],
        "Stock Depletion Rate": np.random.uniform(5, 20, 7),
        "Foot Traffic Index": np.random.uniform(100, 500, 7)
    })
    
    fig1 = px.line(hist_data, x="Date", y=["Stock Depletion Rate", "Foot Traffic Index"], 
                   title="7-Day Velocity & Traffic Correlation", template="plotly_dark")
    st.plotly_chart(fig1, use_container_width=True)
    
    # 7. Inventory Heatmap
    if 'df' in locals():
        st.markdown("### Current Shelf Density Heatmap")
        fig2 = px.treemap(df, path=['Status', 'Product'], values='Qty', 
                          color='Qty', color_continuous_scale='Viridis', template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Run a scan in the Live Scanner tab to generate heatmaps.")

with tab3:
    st.subheader("Raw JSON Telemetry Logs")
    if 'res' in locals():
        st.download_button("Export Session JSON", data=res.tojson(), file_name="telemetry_log.json")
        st.json(res.tojson())
