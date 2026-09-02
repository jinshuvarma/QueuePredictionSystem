import streamlit as st
import json
import time
import os

# Set page config for fullscreen look
st.set_page_config(page_title="Live Queue Signage", layout="wide", initial_sidebar_state="collapsed")

# Read the shared data from the main dashboard
try:
    if os.path.exists("shared_state.json"):
        with open("shared_state.json", "r") as f:
            data = json.load(f)
    else:
        data = {"message": "INITIALIZING SYSTEM...", "color": "#111"}
except Exception:
    data = {"message": "READING SENSOR DATA...", "color": "#111"}

# Aggressive CSS to hide all Streamlit elements and make it look like a physical TV screen
st.markdown(
    f"""
    <style>
        /* Hide Header, Sidebar, and Footer */
        [data-testid="stHeader"] {{ display: none !important; }}
        [data-testid="stSidebar"] {{ display: none !important; }}
        [data-testid="collapsedControl"] {{ display: none !important; }}
        footer {{ display: none !important; }}
        
        /* Fullscreen background color */
        .stApp {{
            background-color: {data['color']};
            display: flex;
            justify-content: center;
            align-items: center;
            text-align: center;
            height: 100vh;
        }}
        
        /* Massive typography for the billboard */
        .signage-text {{
            font-size: 5rem;
            font-weight: 900;
            color: white;
            text-transform: uppercase;
            line-height: 1.2;
            padding: 40px;
            text-shadow: 2px 2px 10px rgba(0,0,0,0.5);
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Display the message
st.markdown(f'<div class="signage-text">{data["message"]}</div>', unsafe_allow_html=True)

# Refresh exactly every 1 second to stay synced with the AI dashboard
time.sleep(1)
st.rerun()