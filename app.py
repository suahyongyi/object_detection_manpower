import streamlit as st
import pandas as pd
import time
from datetime import datetime
import numpy as np
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="Dynamic Resourcing Prototype", layout="wide")

# GITHUB CONFIGURATION
BASE_REPO_URL = "https://raw.githubusercontent.com/suahyongyi/object_detection_manpower/main/simulation_data"
TOTAL_IMAGES_IN_FOLDER = 10 

@st.cache_resource
def load_model():
    return YOLO('best.pt') 

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure 'best.pt' is in the directory.")
    st.stop()

# Initialize Session State
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'history_data' not in st.session_state:
    st.session_state.history_data = []
if 'frame_index' not in st.session_state:
    st.session_state.frame_index = 0

# --- 2. SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Settings")
min_staff_threshold = st.sidebar.number_input("Minimum Staff Count", min_value=0, value=1)
cust_per_staff_threshold = st.sidebar.number_input("Max Customers per Staff", min_value=1, value=2)

st.sidebar.divider()
st.sidebar.header("📡 Stream Controls")
col_start, col_stop = st.sidebar.columns(2)

if col_start.button("🟢 START"):
    st.session_state.is_running = True
    st.rerun()

if col_stop.button("🔴 STOP"):
    st.session_state.is_running = False
    st.rerun()

# --- 3. HELPER FUNCTIONS ---
def fetch_image_from_github(location_folder, filename):
    url = f"{BASE_REPO_URL}/{location_folder}/{filename}"
    try:
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            return None
    except:
        return None

def process_frame(image):
    results = model.predict(image, conf=0.25, verbose=False, agnostic_nms=True)
    boxes = results[0].boxes
    staff_count = int((boxes.cls == 1).sum()) 
    cust_count = int((boxes.cls == 0).sum())
    total_count = staff_count + cust_count
    
    if staff_count == 0:
        ratio = float('inf') if cust_count > 0 else 0.0
    else:
        ratio = round(cust_count / staff_count, 2)
        
    return results[0].plot(), staff_count, cust_count, total_count, ratio

def generate_alert_status(staff, ratio, min_staff, max_ratio):
    alerts = []
    if staff < min_staff:
        alerts.append("Low Staff")
    if ratio > max_ratio:
        alerts.append("High Ratio")
    return ", ".join(alerts) if alerts else ""

# --- 4. MAIN LAYOUT ---
tab1, tab2 = st.tabs(["🎥 Surveillance", "📜 History"])

# --- TAB 1: SURVEILLANCE ---
with tab1:
    # --- TOP ROW LAYOUT: Header | Status | Timer ---
    # We create 3 columns to hold the elements side-by-side
    c_head, c_status, c_timer = st.columns([2.5, 0.8, 1.2])
    
    with c_head:
        st.subheader("Live Operational View")
        # Timestamp sits neatly under the title
        timestamp_placeholder = st.empty() 
        
    with c_status:
        st.write("") # Vertical spacer to align with header
        status_placeholder = st.empty()
        
    with c_timer:
        st.write("") # Vertical spacer
        # The Progress Bar is now here (Top Right)
        progress_bar = st.empty()

    st.divider()
    
    # --- GRID LAYOUT: 2x2 ---
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    
    loc_placeholders = [] 
    grid_columns = [row1_col1, row1_col2, row2_col1, row2_col2]
    
    for i, col in enumerate(grid_columns):
        with col:
            st.markdown(f"**Location {i+1}**")
            c_img, c_txt = st.columns([1, 1.2]) 
            ph_img = c_img.empty()
            ph_txt = c_txt.empty()
            loc_placeholders.append((ph_img, ph_txt))
            st.divider()

# --- TAB 2: HISTORY ---
with tab2:
    st.subheader("Event Log")
    history_table_placeholder = st.empty()

# --- 5. THE MAIN LOOP ---
if st.session_state.is_running:
    
    UPDATE_INTERVAL = 30 
    next_update_time = time.time()
    
    while st.session_state.is_running:
        
        current_time = time.time()
        time_left = next_update_time - current_time
        
        if time_left <= 0:
            # --- PROCESS BATCH ---
            status_placeholder.info("⚡ PROCESSING...")
            progress_bar.progress(0, text="Fetching...")
            
            # 1. Update Time (Left Aligned now looks better in the header column)
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            timestamp_placeholder.markdown(f"<span style='color:gray'>🕒 {now_str}</span>", unsafe_allow_html=True)
            
            # 2. Fetch Images
            idx = (st.session_state.frame_index % TOTAL_IMAGES_IN_FOLDER) + 1
            filename = f"frame_{idx:03d}.jpg" 
            
            locations = ["loc1", "loc2", "loc3", "loc4"]
            batch_images = []
            for loc in locations:
                batch_images.append(fetch_image_from_github(loc, filename))
            
            if all(batch_images):
                new_records_batch = []
                
                for i, img in enumerate(batch_images):
                    ph_img, ph_txt = loc_placeholders[i]
                    plotted_img, staff, cust, total, ratio = process_frame(img)
                    
                    # Alerts
                    staff_alert = "🚨" if staff < min_staff_threshold else ""
                    ratio_alert = "🚨" if (ratio == float('inf') or ratio > cust_per_staff_threshold) else ""
                    ratio_display = "∞" if ratio == float('inf') else str(ratio)
                    
                    # Update UI
                    ph_img.image(plotted_img[..., ::-1], channels="RGB", width=250)
                    
                    metrics_html = f"""
                    <div style="font-size: 0.9rem; line-height: 1.4;">
                        <b>Total:</b> {total}<br>
                        <b>Staff:</b> {staff} <span style="color:red">{staff_alert}</span><br>
                        <b>Cust:</b> {cust}<br>
                        <b>Ratio:</b> {ratio_display} <span style="color:red">{ratio_alert}</span>
                    </div>
                    """
                    ph_txt.markdown(metrics_html, unsafe_allow_html=True)
                    
                    # History Record
                    loc_name = f"Location {i+1}"
                    alert_msg = generate_alert_status(staff, ratio, min_staff_threshold, cust_per_staff_threshold)
                    new_records_batch.append({
                        "DateTime": now_str,
                        "Location": loc_name,
                        "Total": total,
                        "Staff": staff,
                        "Customer": cust,
                        "Ratio": ratio_display,
                        "Alerts": alert_msg
                    })

                # Update History
                for record in reversed(new_records_batch):
                    st.session_state.history_data.insert(0, record)

                history_table_placeholder.dataframe(
                    pd.DataFrame(st.session_state.history_data), 
                    use_container_width=True,
                    height=400,
                    column_config={"Alerts": st.column_config.TextColumn("Alerts")}
                )

                status_placeholder.success(f"✅ ACTIVE (Batch {idx})")
                st.session_state.frame_index += 1
                next_update_time = time.time() + UPDATE_INTERVAL
            
            else:
                status_placeholder.error(f"❌ Failed: {filename}")
                time.sleep(5)

        else:
            # --- IDLE COUNTDOWN ---
            time.sleep(0.1)
            percent = 1 - (time_left / UPDATE_INTERVAL)
            percent = max(0.0, min(1.0, percent))
            progress_bar.progress(percent, text=f"Next scan in {int(time_left)}s")

else:
    status_placeholder.warning("🛑 OFFLINE")
    if st.session_state.history_data:
        history_table_placeholder.dataframe(
            pd.DataFrame(st.session_state.history_data), 
            use_container_width=True,
            column_config={"Alerts": st.column_config.TextColumn("Alerts")}
        )