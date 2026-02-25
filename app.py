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

# 🔗 GITHUB CONFIGURATION
# We use the 'raw' URL to fetch images directly from your repo
BASE_REPO_URL = "https://raw.githubusercontent.com/suahyongyi/object_detection_manpower/main/simulation_data"

# ⚠️ IMPORTANT: Update this to match your actual total image count per folder
TOTAL_IMAGES_IN_FOLDER = 10 

@st.cache_resource
def load_model():
    # Load your custom trained model
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
# We store the frame_index in session state so it persists across reruns
if 'frame_index' not in st.session_state:
    st.session_state.frame_index = 0

# --- 2. SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Settings")
min_staff_threshold = st.sidebar.number_input("Minimum Staff Count", min_value=0, value=1)
cust_per_staff_threshold = st.sidebar.number_input("Max Customers per Staff", min_value=1, value=2)

st.sidebar.divider()
st.sidebar.header("📡 Stream Controls")
col_start, col_stop = st.sidebar.columns(2)

# BUTTON LOGIC: We use 'st.rerun()' to make the click instant
if col_start.button("🟢 START"):
    st.session_state.is_running = True
    st.rerun()

if col_stop.button("🔴 STOP"):
    st.session_state.is_running = False
    st.rerun()

# --- 3. HELPER FUNCTIONS ---
def fetch_image_from_github(location_folder, filename):
    """
    Fetches an image from your specific GitHub repo structure.
    """
    # Construct URL: .../simulation_data/loc1/frame_000.jpg
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
    """Run inference and calculate metrics for a single image."""
    results = model.predict(image, conf=0.25, verbose=False, agnostic_nms=True)
    
    # Check your class IDs! Assuming 0=Customer, 1=Staff
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
    col_header, col_status = st.columns([3, 1])
    col_header.subheader("Live Operational View")
    
    # Status Indicators
    status_placeholder = col_status.empty()
    timestamp_placeholder = st.empty()
    
    # Create 4 Rows for 4 Locations
    loc_containers = []
    for i in range(4):
        with st.container():
            st.markdown(f"**Location {i+1}**")
            c1, c2 = st.columns([2, 1]) 
            loc_containers.append((c1, c2)) # Store empty columns to fill later
            st.divider()

    # --- THE RESPONSIVE LOOP ---
    if st.session_state.is_running:
        
        # We use a placeholder for the countdown bar
        progress_bar = st.empty()
        
        # State variables for the loop
        UPDATE_INTERVAL = 30  # Seconds
        next_update_time = time.time() # Start immediately
        
        # MAIN APP LOOP
        while st.session_state.is_running:
            
            # 1. CHECK TIMER: Is it time to run a batch?
            current_time = time.time()
            time_left = next_update_time - current_time
            
            if time_left <= 0:
                # --- ACTION: RUN BATCH PROCESSING ---
                status_placeholder.info("⚡ PROCESSING BATCH...")
                progress_bar.progress(0, text="Fetching images from GitHub...")
                
                # A. Update Timestamp
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                timestamp_placeholder.markdown(f"<h3 style='text-align: right;'>🕒 {now_str}</h3>", unsafe_allow_html=True)
                
                # B. Generate Filename (Cycling 0 to TOTAL_IMAGES)
                # Assumes filename format: frame_000.jpg, frame_001.jpg
                idx = st.session_state.frame_index % TOTAL_IMAGES_IN_FOLDER
                filename = f"frame_{idx:03d}.jpg" 
                
                # C. Fetch & Process for all 4 Locations
                locations = ["loc1", "loc2", "loc3", "loc4"]
                batch_images = []
                
                for loc in locations:
                    img = fetch_image_from_github(loc, filename)
                    batch_images.append(img)
                
                # Only proceed if we successfully fetched at least one image
                # (We use 'all' to be strict, or 'any' to be lenient)
                if all(batch_images):
                    for i, img in enumerate(batch_images):
                        loc_name = f"Location {i+1}"
                        col_img, col_metrics = loc_containers[i]
                        
                        plotted_img, staff, cust, total, ratio = process_frame(img)
                        
                        # Display Image
                        col_img.image(plotted_img[..., ::-1], channels="RGB", use_container_width=True)
                        
                        # Display Metrics
                        staff_alert = "🚨" if staff < min_staff_threshold else ""
                        ratio_alert = "🚨" if (ratio == float('inf') or ratio > cust_per_staff_threshold) else ""
                        ratio_display = "∞" if ratio == float('inf') else str(ratio)
                        
                        metrics_html = f"""
                        <div style="font-size: 1.1rem; line-height: 1.6;">
                            <b>Total:</b> {total}<br>
                            <b>Staff:</b> {staff} <span style="color:red">{staff_alert}</span><br>
                            <b>Customer:</b> {cust}<br>
                            <b>Cust/Staff:</b> {ratio_display} <span style="color:red">{ratio_alert}</span>
                        </div>
                        """
                        col_metrics.markdown(metrics_html, unsafe_allow_html=True)
                        
                        # Log History
                        alert_msg = generate_alert_status(staff, ratio, min_staff_threshold, cust_per_staff_threshold)
                        new_record = {
                            "DateTime": now_str,
                            "Location": loc_name,
                            "Total": total,
                            "Staff": staff,
                            "Customer": cust,
                            "Ratio": ratio_display,
                            "Alerts": alert_msg
                        }
                        st.session_state.history_data.insert(0, new_record)
                    
                    status_placeholder.success(f"✅ BATCH {idx} COMPLETE")
                
                else:
                    status_placeholder.error(f"❌ Failed to fetch batch: {filename}")
                
                # D. Advance Counter & Reset Timer
                st.session_state.frame_index += 1
                next_update_time = time.time() + UPDATE_INTERVAL
                
            else:
                # --- IDLE STATE: UPDATE COUNTDOWN ---
                # Calculate percentage for progress bar (0% at 30s, 100% at 0s)
                percent_complete = 1 - (time_left / UPDATE_INTERVAL)
                percent_complete = max(0.0, min(1.0, percent_complete)) # Clamp between 0 and 1
                
                progress_bar.progress(percent_complete, text=f"Next scan in {int(time_left)} seconds...")
                
                # IMPORTANT: Small sleep to prevent CPU burning
                # This keeps the loop running but allows Streamlit to catch "Stop" clicks
                time.sleep(0.1) 

    else:
        status_placeholder.warning("🛑 SYSTEM OFFLINE")

# --- TAB 2: HISTORY ---
with tab2:
    st.subheader("Event Log")
    if st.session_state.history_data:
        df_history = pd.DataFrame(st.session_state.history_data)
        st.dataframe(
            df_history, 
            use_container_width=True,
            column_config={
                "Alerts": st.column_config.TextColumn("Alerts", help="Triggered threshold violations")
            }
        )
    else:
        st.info("No data yet. Start the stream to collect logs.")