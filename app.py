import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from PIL import Image
from ultralytics import YOLO
from datetime import datetime
import random

# 1. Enterprise UI & Dark Mode CSS
st.set_page_config(page_title="StockSense Pro | Enterprise Retail AI", page_icon="🛒", layout="wide")
st.markdown("""
    <style>
    .main {background-color: #0b0f19;}
    h1, h2, h3 {color: #00e5ff;}
    .stMetric {background-color: #1a2235; padding: 15px; border-radius: 8px; border-left: 4px solid #00e5ff;}
    .task-card {background-color: #2a0a18; padding: 10px; border-left: 4px solid #ff4b4b; margin-bottom: 10px;}
    </style>
""", unsafe_allow_html=True)

# 2. System Header
st.title("🛒 StockSense Pro // ENTERPRISE COMMAND CENTER")
st.caption(f"SYSTEM ONLINE | Node: Madurai-01 | Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Env: Production")

# 3. Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Neural Engine Specs")
    conf_threshold = st.slider("AI Confidence Threshold", 0.1, 1.0, 0.40, 0.05)
    iou_threshold = st.slider("NMS Overlap Threshold", 0.1, 1.0, 0.50, 0.05)
    st.markdown("---")
    st.header("🏪 Store Operations")
    shift_manager = st.selectbox("Active Shift Manager", ["Sarah Jenkins", "Michael Chen", "David Kumar"])
    target_compliance = st.slider("Target Planogram Compliance %", 50, 100, 85)

# 4. Load AI Model
@st.cache_resource
def load_vision_model():
    return YOLO("best.pt")

try:
    model = load_vision_model()
except Exception as e:
    st.error(f"SYSTEM HALT: Weights 'best.pt' missing. ({e})")
    st.stop()

# 5. The Retail SKU Translation Engine (Makes it look realistic)
REALISTIC_BRANDS = {
    "bottle": ["Coca-Cola 500ml", "Pepsi Max 500ml", "Sprite Zero"],
    "cup": ["Red Bull 250ml", "Monster Energy"],
    "book": ["Kellogg's Corn Flakes", "Oreo Family Pack"],
    "apple": ["Fresh Apples (Fuji)"],
    "banana": ["Fresh Bananas (Cavendish)"],
    "orange": ["Florida Oranges"],
    "snack": ["Lay's Classic", "Doritos Nacho Cheese"]
}

def translate_to_sku(raw_class):
    # Maps generic AI classes to realistic brands
    if raw_class in REALISTIC_BRANDS:
        return random.choice(REALISTIC_BRANDS[raw_class])
    return raw_class.capitalize()

# 6. Multi-Tab Architecture
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📷 Live Scanner", "📊 Planogram & Health", "💰 Financial Analytics", "📋 Staff Operations", "🗄️ System Logs"
])

# Initialize session state for data persistence
if 'inventory_df' not in st.session_state:
    st.session_state.inventory_df = None

with tab1:
    col_input, col_output = st.columns([1, 1.5])
    
    with col_input:
        st.subheader("Video Telemetry Feed")
        img_source = st.file_uploader("Upload High-Res Aisle Scan", type=["jpg", "png", "jpeg"])
        
    if img_source:
        image = Image.open(img_source).convert("RGB")
        
        with st.spinner("Executing YOLOv8 Neural Network..."):
            res = model.predict(image, conf=conf_threshold, iou=iou_threshold)[0]
            
        with col_output:
            st.subheader("AI Bounding Box Overlay")
            st.image(res.plot(), use_container_width=True)
            
        # Process Detections and Apply Realistic Brands
        names = model.names
        raw_detected = [names[int(b.cls[0])] for b in res.boxes]
        
        # Translate to realistic SKUs
        sku_list = [translate_to_sku(item) for item in raw_detected]
        inventory = {item: sku_list.count(item) for item in set(sku_list)}
        
        # Build Operations DataFrame
        df = pd.DataFrame(list(inventory.items()), columns=["SKU", "Units on Shelf"])
        df["Target Stock"] = np.random.randint(10, 25, size=len(df)) # Simulated planogram targets
        df["Unit Price ($)"] = np.random.uniform(1.99, 7.99, size=len(df)).round(2)
        df["Status"] = np.where(df["Units on Shelf"] == 0, "Critical", np.where(df["Units on Shelf"] < df["Target Stock"] * 0.4, "Warning", "Optimal"))
        
        st.session_state.inventory_df = df
        st.session_state.raw_json = res.tojson()

        st.markdown("### ⚡ Live Telemetry Readout")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Units Tracked", len(sku_list))
        m2.metric("Unique SKUs", len(df))
        m3.metric("Critical Shortages", len(df[df['Status']=='Critical']), delta="Urgent Action Required", delta_color="inverse")
        m4.metric("Inference Latency", f"{round(res.speed['inference'], 2)} ms")

if st.session_state.inventory_df is not None:
    df = st.session_state.inventory_df
    
    with tab2:
        st.subheader("Planogram Compliance & Heatmap")
        
        # Calculate Compliance
        total_current = df["Units on Shelf"].sum()
        total_target = df["Target Stock"].sum()
        compliance = round((total_current / total_target) * 100, 1) if total_target > 0 else 100
        
        col1, col2 = st.columns(2)
        col1.metric("Current Planogram Compliance", f"{compliance}%", delta=f"{round(compliance - target_compliance, 1)}% vs Target")
        
        if compliance < target_compliance:
            st.error(f"⚠️ Compliance below {target_compliance}%. Restock required to maintain brand agreements.")
        else:
            st.success("✅ Shelf meets visual merchandising standards.")
            
        fig_tree = px.treemap(df, path=['Status', 'SKU'], values='Units on Shelf', 
                              color='Units on Shelf', color_continuous_scale='Magma', 
                              title="Inventory Density Heatmap", template="plotly_dark")
        st.plotly_chart(fig_tree, use_container_width=True)

    with tab3:
        st.subheader("Financial Impact & ROI")
        
        # Financial Calculations
        df["Lost Revenue Potential"] = ((df["Target Stock"] - df["Units on Shelf"]).clip(lower=0) * df["Unit Price ($)"]).round(2)
        total_at_risk = df["Lost Revenue Potential"].sum()
        
        st.metric("Total Revenue at Risk (Out of Stock)", f"${total_at_risk}")
        
        fig_bar = px.bar(df, x="SKU", y="Lost Revenue Potential", color="Status",
                         color_discrete_map={"Critical": "#ff4b4b", "Warning": "#ffa500", "Optimal": "#00FFCC"},
                         title="Revenue at Risk by SKU", template="plotly_dark")
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab4:
        st.subheader("📋 Automated Task Assignments")
        st.markdown(f"**Manager on Duty:** {shift_manager}")
        
        urgent_tasks = df[df["Status"] != "Optimal"]
        if urgent_tasks.empty:
            st.success("No urgent restock tasks generated. Floor is optimal.")
        else:
            for index, row in urgent_tasks.iterrows():
                shortage = row["Target Stock"] - row["Units on Shelf"]
                st.markdown(f"""
                <div class="task-card">
                    <strong>Urgent Restock:</strong> {row['SKU']}<br>
                    Action: Bring <strong>{shortage} units</strong> from backroom to Aisle 4.<br>
                    <em>Revenue blocked: ${row['Lost Revenue Potential']}</em>
                </div>
                """, unsafe_allow_html=True)

    with tab5:
        st.subheader("System Logs & Data Export")
        st.dataframe(df, use_container_width=True)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Compliance Report (CSV)", data=csv, file_name=f"Compliance_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
        
        with st.expander("View Raw AI Telemetry (JSON)"):
            st.json(st.session_state.raw_json)
else:
    for t in [tab2, tab3, tab4, tab5]:
        with t:
            st.info("System Standby: Upload an image in the Live Scanner tab to generate analytics.")
